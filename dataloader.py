from utils.utils import *



# label_dict = {'transfer':'banking', 'transactions':'banking', 'balance':'banking', 'freeze_account':'banking', 'pay_bill':'banking', 'bill_balance':'banking', 'bill_due':'banking', 'interest_rate':'banking', 'routing':'banking', 'min_payment':'banking', 'order_checks':'banking', 'pin_change':'banking', 'report_fraud':'banking', 'account_blocked':'banking', 'spending_history':'banking', 
# 'credit_score':'credit_cards', 'report_lost_card':'credit_cards', 'credit_limit':'credit_cards', 'rewards_balance':'credit_cards', 'new_card':'credit_cards', 'application_status':'credit_cards', 'card_declined':'credit_cards', 'international_fees':'credit_cards', 'apr':'credit_cards', 'redeem_rewards':'credit_cards', 'credit_limit_change':'credit_cards', 'damaged_card':'credit_cards', 'replacement_card_duration':'credit_cards', 'improve_credit_score':'credit_cards', 'expiration_date':'credit_cards', 
# 'recipe':'kitchen', 'restaurant_reviews':'kitchen', 'calories':'kitchen', 'nutrition_info':'kitchen', 'restaurant_suggestion':'kitchen', 'ingredients_list':'kitchen', 'ingredient_substitution':'kitchen', 'cook_time':'kitchen', 'food_last':'kitchen', 'meal_suggestion':'kitchen', 'restaurant_reservation':'kitchen', 'confirm_reservation':'kitchen', 'how_busy':'kitchen', 'cancel_reservation':'kitchen', 'accept_reservations':'kitchen', 
# 'shopping_list':'home', 'shopping_list_update':'home', 'next_song':'home', 'play_music':'home', 'update_playlist':'home', 'todo_list':'home', 'todo_list_update':'home', 'calendar':'home', 'calendar_update':'home', 'what_song':'home', 'order':'home', 'order_status':'home', 'reminder':'home', 'reminder_update':'home', 'smart_home':'home', 
# 'traffic':'auto', 'directions':'auto', 'gas':'auto', 'gas_type':'auto', 'distance':'auto', 'current_location':'auto', 'mpg':'auto', 'oil_change_when':'auto', 'oil_change_how':'auto', 'jump_start':'auto', 'uber':'auto', 'schedule_maintenance':'auto', 'last_maintenance':'auto', 'tire_pressure':'auto', 'tire_change':'auto', 
# 'book_flight':'travel', 'book_hotel':'travel', 'car_rental':'travel', 'travel_suggestion':'travel', 'travel_alert':'travel', 'travel_notification':'travel', 'carry_on':'travel', 'timezone':'travel', 'vaccines':'travel', 'translate':'travel', 'flight_status':'travel', 'international_visa':'travel', 'lost_luggage':'travel', 'plug_type':'travel', 'exchange_rate':'travel', 
# 'time':'utility', 'alarm':'utility', 'share_location':'utility', 'find_phone':'utility', 'weather':'utility', 'text':'utility', 'spelling':'utility', 'make_call':'utility', 'timer':'utility', 'date':'utility', 'calculator':'utility', 'measurement_conversion':'utility', 'flip_coin':'utility', 'roll_dice':'utility', 'definition':'utility', 
# 'direct_deposit':'work', 'pto_request':'work', 'taxes':'work', 'payday':'work', 'w2':'work', 'pto_balance':'work', 'pto_request_status':'work', 'next_holiday':'work', 'insurance':'work', 'insurance_change':'work', 'schedule_meeting':'work', 'pto_used':'work', 'meeting_schedule':'work', 'rollover_401k':'work', 'income':'work', 
# 'greeting':'talk', 'goodbye':'talk', 'tell_joke':'talk', 'where_are_you_from':'talk', 'how_old_are_you':'talk', 'what_is_your_name':'talk', 'who_made_you':'talk', 'thank_you':'talk', 'what_can_i_ask_you':'talk', 'what_are_your_hobbies':'talk', 'do_you_have_pets':'talk', 'are_you_a_bot':'talk', 'meaning_of_life':'talk', 'who_do_you_work_for':'talk', 'fun_fact':'talk', 
# 'change_ai_name':'meta', 'change_user_name':'meta', 'cancel':'meta', 'user_name':'meta', 'reset_settings':'meta', 'whisper_mode':'meta', 'repeat':'meta', 'no':'meta', 'yes':'meta', 'maybe':'meta', 'change_language':'meta', 'change_accent':'meta', 'change_volume':'meta', 'change_speed':'meta', 'sync_device':'meta'}

class Data:

    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'hwu': 20, 'clinc': 30, 'stackoverflow': 45, 'banking': 65}
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.known_lab = [i for i in range(len(self.known_label_list))]

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples(processor, args, 'train')
        print('num_labeled_samples', len(self.train_labeled_examples))
        print('num_unlabeled_samples', len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args, 'train')
        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.get_semi(
            self.train_labeled_examples, self.train_unlabeled_examples, args)
        self.train_semi_dataset, self.train_semi_dataloader = self.get_semi_loader(self.semi_input_ids, self.semi_input_mask,
                                                                self.semi_segment_ids, self.semi_label_ids, args)
        self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_examples, args, 'test')

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])
                train_labeled_ids.extend(random.sample(pos, num))

            train_labeled_examples, train_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples, args):

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        labeled_features = convert_examples_to_features(labeled_examples, self.known_label_list, args.max_seq_length,
                                                        tokenizer)
        unlabeled_features = convert_examples_to_features_semi(unlabeled_examples, self.known_label_list, 
                                                               self.all_label_list, args.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        # unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = RandomSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size=args.train_batch_size)

        return semi_data, semi_dataloader

    def get_loader(self, examples, args, mode='train'):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        # char → id
        if mode == 'train' or mode == 'eval':
            features = convert_examples_to_features(examples, self.known_label_list, args.max_seq_length, tokenizer)
        elif mode == 'test':
            features = convert_examples_to_features_semi(examples, self.known_label_list, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

        return dataloader


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        if data_dir == 'data/banking':
            labels = np.unique(np.array(test['label']))
        else:
            labels = [str(label).lower() for label in test['label']]
            labels = np.unique(np.array(labels))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def convert_examples_to_features_semi(examples, labeled_list, all_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(labeled_list):
        label_map[label] = i
    
    novel_list = [label for label in all_list if label not in labeled_list]
    for i, label in enumerate(novel_list):
        label_map[label] = i + len(labeled_list)
    
    # label_list = []
    # if len(label_map) == 150:
    #     for i in label_dict.keys():
    #         label_list.append(label_map[i])
    #         if len(label_list) == 15:
    #             print(label_list)
    #             label_list = []

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()

    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
