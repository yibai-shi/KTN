from utils.utils import *
from utils.sinkhorn_knopp import *
from model import *
from dataloader import *

import ot
import math
import time
import json
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn import mixture


class Manager:

    def __init__(self, args, data, pretrained_model):
        set_seed(args.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BertForOT(args.bert_model, num_labels=data.num_labels)
        self.model.to(self.device)
        self.pretrained_model = pretrained_model
        self.load_pretrained_model()
        # decoupled prototypes for classifier initialization
        self.initialize_classifier(args, data)
        # updating last four layers while keeping other parameters frozen
        self.freeze_parameters(self.model)

        self.num_train_optimization_steps = int(len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args)

        # Data Augmentation -> random token replace
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

        # logits -> utils.sinkhorn_knopp -> pseudo-label
        self.sinkhorn = SinkhornKnopp(args)

    def train(self, args, data):

        unlabeled_iter = iter(data.train_semi_dataloader)

        for epoch in range(int(args.num_train_epochs)):

            print('---------------------------')
            print(f'training epoch:{epoch}')

            self.model.train()

            # compute transfer weights between known and unknown categories during the whole training process only once
            if epoch == 0:
                optimal_map, _ = self.ot_kt(data)

            # The threshold that determines whether pseudo-labels of unknown categories needs to be converted to one-hot labels
            threshold = args.threshold

            for batch in tqdm(data.train_labeled_dataloader, desc="Pseudo-label training"):

                # acquire labeled data
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch
                X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                labels = torch.zeros(len(labels), data.num_labels, device=self.device).scatter_(1, labels.view(-1,1).long(), 1)

                # acquire unlabeled data / data augmentation
                try:
                    batch_u = unlabeled_iter.next()
                    batch_u = tuple(t.to(self.device) for t in batch_u)
                except StopIteration:
                    unlabeled_iter = iter(data.train_semi_dataloader)
                    batch_u = unlabeled_iter.next()
                    batch_u = tuple(t.to(self.device) for t in batch_u)
                input_ids, input_mask, segment_ids, _ = batch_u
                X_u1 = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                X_u2 = {"input_ids": self.generator.random_token_replace(input_ids.cpu()).to(self.device), 
                        "attention_mask": input_mask, "token_type_ids": segment_ids}
                
                # compute entropy with pre-trained model
                with torch.no_grad():
                    _, logits = self.pretrained_model(X_u1, output_hidden_states=True)
                    probs = torch.softmax(logits, dim=1)
                    entropy = torch.sum(-probs*torch.log(probs), dim=1)
                    entropy_mean = torch.mean(entropy)

                # weight normalization of prototype classifier
                with torch.no_grad():
                    w = self.model.classifier.weight.data.clone()
                    w = F.normalize(w, dim=1, p=2)
                    self.model.classifier.weight.copy_(w)

                with torch.no_grad():
                    # logits adjustment
                    _, logits_u1 = self.model(X_u1)
                    self.logits_adjustment(data, logits_u1, optimal_map, entropy, entropy_mean)
                    _, logits_u2 = self.model(X_u2)
                    self.logits_adjustment(data, logits_u2, optimal_map, entropy, entropy_mean)

                    # pseudo-label assignment with adjusted logits
                    labels_u1 = self.sinkhorn(logits_u2)
                    labels_u2 = self.sinkhorn(logits_u1)

                # select high confidence novel category example based on paeudo-label
                hard_novel_idx1, soft_index1 = self.split_hard_novel_soft_seen(data, labels_u1, threshold)
                hard_novel_idx2, soft_index2 = self.split_hard_novel_soft_seen(data, labels_u2, threshold)

                # label adjustment similar as logits adjustment
                self.labels_adjustment(data, labels_u1, optimal_map, entropy, entropy_mean, soft_index1)
                self.labels_adjustment(data, labels_u2, optimal_map, entropy, entropy_mean, soft_index2)

                # convert soft labels of high confidence novel category examples to one-hot labels
                self.gen_hard_novel(labels_u1, hard_novel_idx1, threshold)
                self.gen_hard_novel(labels_u2, hard_novel_idx2, threshold)

                X_u = {"input_ids": torch.cat([X_u1["input_ids"], X_u2["input_ids"]], dim=0), 
                       "attention_mask": torch.cat([X_u1["attention_mask"], X_u2["attention_mask"]], dim=0),
                       "token_type_ids": torch.cat([X_u1["token_type_ids"], X_u2["token_type_ids"]], dim=0)}
                labels_u = torch.cat([labels_u1, labels_u2], dim=0)

                _, logits_l = self.model(X)
                feats_u, logits_u = self.model(X_u)
                feats_u1 = feats_u[:len(labels_u1), :]
                feats_u2 = feats_u[len(labels_u1):, :]
                logits_l = F.normalize(logits_l, dim=1)
                logits_u = F.normalize(logits_u, dim=1)

                loss_cel = -torch.mean(torch.sum(labels * F.log_softmax((logits_l), dim=1), dim=1))
                loss_ceu = -torch.mean(torch.sum(labels_u * F.log_softmax((logits_u), dim=1), dim=1))
                loss_contrast = self.model.loss_contrast(feats_u1, feats_u2, 0.07)
                loss = 0.7 * loss_ceu + 0.3 * loss_cel + 0.01 * loss_contrast

                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    def initialize_classifier(self, args, data):
        # extract labeled prototypes
        feats, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        [rows, _] = feats.shape
        num = np.zeros(data.n_known_cls)
        self.proto_l = np.zeros((data.n_known_cls, args.feat_dim))
        for i in range(rows):
            self.proto_l[labels[i]] += feats[i]
            num[labels[i]] += 1
        for i in range(data.n_known_cls):
            self.proto_l[i] = self.proto_l[i] / num[i]

        # extract and rank unlabeled prototypes
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels, n_init=20).fit(feats)
        self.proto_u = km.cluster_centers_
        distance = dist.cdist(self.proto_l, self.proto_u, 'euclidean')
        _, col_ind = linear_sum_assignment(distance)
        pro_l = []
        for i in range(len(col_ind)):
            pro_l.append(self.proto_u[col_ind[i]][:])
        pro_u = []
        for j in range(data.num_labels):
            if j not in col_ind:
                pro_u.append(self.proto_u[j][:])
        self.proto_u = pro_l + pro_u   
        self.proto_u = torch.tensor(np.array(self.proto_u), dtype=torch.float).to(self.device)

        # initialize prototype classifier
        self.model.classifier.weight.data = self.proto_u
        with torch.no_grad():
            w = self.model.classifier.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.model.classifier.weight.copy_(w)

    def ot_kt(self, data):
        pro_u = self.proto_u[len(self.proto_l):].cpu()
        optimal_map = F.normalize(torch.tensor(self.proto_l, dtype=torch.float), dim=1) @ F.normalize(torch.tensor(pro_u, dtype=torch.float).t(), dim=0)
        optimal_map = F.softmax(optimal_map)
        optimal_map = optimal_map.to(self.device)

        padding = torch.zeros(data.n_known_cls, data.n_known_cls, dtype=torch.float).to(self.device)
        optimal_map_padding = torch.cat((padding, optimal_map), 1)

        return optimal_map, optimal_map_padding

    def logits_adjustment(self, data, logits, optimal_map, entropy, entropy_mean):
        # determine differentiation probability
        mask = torch.sigmoid(((entropy - entropy_mean) - 1 * torch.max(entropy - entropy_mean)))
        mask = mask.reshape(-1, 1)

        logits_seen = logits[:, :data.n_known_cls]
        cand_values, cand_index = torch.topk(logits_seen, 1)

        # adjust logits
        logits_transfer = torch.zeros(logits_seen.shape, device=self.device)
        logits_transfer = logits_transfer.scatter(1, cand_index, cand_values)
        logits[:, :data.n_known_cls] -= mask * torch.abs(logits_transfer)
        logits[:, data.n_known_cls:] += mask * (torch.abs(logits_transfer) @ optimal_map)

    # similar as logits adjustment
    def labels_adjustment(self, data, labels, optimal_map, entropy, entropy_mean, soft_index):
        if soft_index.numel() > 0:
            mask = torch.sigmoid(((entropy - entropy_mean) - 1 * torch.max(entropy - entropy_mean)))
            mask = mask.reshape(-1, 1)
            labels_seen = labels[:, :data.n_known_cls]
            cand_values, cand_index = torch.topk(labels_seen, 1)
            labels_transfer = torch.zeros(labels_seen.shape, device=self.device)
            labels_transfer = labels_transfer.scatter(1, cand_index, cand_values)
            labels[soft_index, :data.n_known_cls] -= mask[soft_index] * labels_transfer[soft_index]
            labels[soft_index, data.n_known_cls:] += mask[soft_index] * (labels_transfer[soft_index] @ optimal_map)

    def split_hard_novel_soft_seen(self, data, labels, threshold):
        labels_novel = labels[:, data.n_known_cls:]
        max_pred_novel, _ = torch.max(labels_novel, dim=-1)
        hard_novel_idx = torch.where(max_pred_novel>=threshold)[0]

        soft_index = torch.tensor([i for i in range(0, len(labels)) if i not in list(hard_novel_idx.cpu().numpy())])

        return hard_novel_idx, soft_index

    def gen_hard_novel(self, labels, hard_novel_idx, threshold):
        labels[hard_novel_idx] = labels[hard_novel_idx].ge(threshold).float()

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler

    def freeze_parameters(self, model):
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            if "encoder.layer.8" in name or "encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation for clustering"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature, _ = model(X, output_hidden_states=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def evaluation(self, data):
        self.model.eval()
        pred_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        feats = torch.empty((0, 768)).to(self.device)

        for batch in data.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feat, logits = self.model(X, output_hidden_states=True)
            labels = torch.argmax(logits, dim=1)

            pred_labels = torch.cat((pred_labels, labels))  
            total_labels = torch.cat((total_labels, label_ids))  
            feats = torch.cat((feats, feat))

        y_pred = pred_labels.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results', results)