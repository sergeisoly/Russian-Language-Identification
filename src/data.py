import io
import os
import re

from torchtext.data import Field, NestedField
import torchtext.data as data
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from typing import Iterable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'

def get_data_fields(fixed_lengths: int) -> dict:
    """"
    Creates torchtext fields for the I/O pipeline.
    """

    language = Field(
        batch_first=True, init_token=None, eos_token=None, pad_token=None, unk_token=None)

    characters = Field(include_lengths=True, batch_first=True, init_token=None,
                       eos_token=END_TOKEN, pad_token=PAD_TOKEN, fix_length=fixed_lengths)

    nesting_field = Field(tokenize=list, pad_token=PAD_TOKEN, batch_first=True,
                          init_token=None, eos_token=END_TOKEN)
    paragraph = NestedField(nesting_field, pad_token=PAD_TOKEN, eos_token=END_TOKEN,
                            include_lengths=True)
    #
    # paragraph = Field(include_lengths=True, batch_first=True, init_token=None,
    #                   eos_token=END_TOKEN, pad_token=PAD_TOKEN)

    fields = {
        'characters': ('characters', characters),
        'paragraph':   ('paragraph', paragraph),
        'language':    ('language', language)
    }

    return fields


def empty_example() -> dict:
    ex = {
        'id':         [],
        'paragraph':  [],
        'language':   [],
        'characters': []
    }
    return ex


def data_reader(x_file: Iterable, y_file: Iterable, train: bool, split_sentences, max_chars: int, level: str) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()
    #spacy_tokenizer = data.get_tokenizer("spacy")  # TODO: implement with word level

    for x, y in zip(x_file, y_file):

        x = x.strip()
        y = y.strip()

        examples = []

        if len(x) == 0: continue
        example = empty_example()

        # replace all numbers with 0
        x = re.sub('[0-9]+', '0', x)
        # x = spacy_tokenizer(x)
        paragraph = x.split()
        if y != 'rus':
            y = 'not_rus'
        language = y

        count = 0

        if level == "char" or train:
            example['paragraph'] = [word.lower() for word in paragraph[:max_chars]]
        else:

            example['paragraph'] = []
            for word in paragraph:
                cur_word = word.lower()
                room_left = max_chars - count
                count += len(cur_word)
                if not count > max_chars and len(cur_word) > 0:
                    example['paragraph'].append(cur_word)
                elif room_left > 0:
                    count -= len(cur_word) + len(''.join(list(cur_word)[:room_left]))
                    example['paragraph'].append(''.join(list(cur_word)[:room_left]))
                    break
                else:
                    count -= len(cur_word)
                    break
            assert count <= max_chars, "too much chars, max_chars: {}, count: {},  room_left: {}".format(max_chars, count, room_left)
            if len(example['paragraph']) == 0:
                continue
        example['language'] = language
        example['characters'] = list(x)[:max_chars]

        examples.append(example)

        yield examples

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield [example]

def test_data_reader(path: str, split_sentences, max_chars: int, level: str) -> dict:
    """
    Return examples as a dictionary.
    """

    example = empty_example()
    #spacy_tokenizer = data.get_tokenizer("spacy")  # TODO: implement with word level
    files = sorted(os.listdir(path))
    for i, file_name in enumerate(files):
        with open(path + file_name, 'r') as f:
            x = f.read()

            x = x.strip()

            examples = []

            if len(x) == 0: continue
            example = empty_example()

            # replace all numbers with 0
            x = re.sub('[0-9]+', '0', x)
            # x = spacy_tokenizer(x)
            paragraph = x.split()

            count = 0

            if level == "char":
                example['paragraph'] = [word.lower() for word in paragraph[:max_chars]]
            else:
                raise NotImplementedError()

            example['characters'] = list(x)[:max_chars]
            example['language'] = 'rus'

            examples.append(example)

        yield examples

    # possible last sentence without newline after
    if len(example['paragraph']) > 0:
        yield [example]


class LanguageDataset(Dataset):

    def sort_key(self, example):
        if self.level == "char":
            return len(example.characters)
        else:
            return len(example.paragraph)

    def __init__(self, paragraph_path: str, label_path: str, taiga_path: str, fields: dict, split_sentences: bool, train: bool,
                 max_chars: int=1000, level: str="char",
                 **kwargs):
        """
        Create a Dataset given a path two the raw text and to the labels and field dict.
        """

        self.level = level
        examples = []

        for d in test_data_reader(taiga_path, split_sentences, max_chars, level):
            for sentence in d:
                # print(sentence)
                # break
                examples.extend([Example.fromdict(sentence, fields)])
                
        with io.open(os.path.expanduser(paragraph_path), encoding="utf8") as f_par, \
                io.open(os.path.expanduser(label_path), encoding="utf8") as f_lab:

            for d in data_reader(f_par, f_lab, train, split_sentences, max_chars, level):
                for sentence in d:
                    # print(sentence)
                    # break
                    examples.extend([Example.fromdict(sentence, fields)])
        

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(LanguageDataset, self).__init__(examples, fields, **kwargs)


def load_data(training_text: str, training_labels: str, testing_text: str, testing_labels: str,
              validation_text: str, validation_labels: str, 
              taiga_train_path: str, taiga_valid_path:str, taiga_test_path: str, 
              max_chars: int=1000, max_chars_test: int=-1,
              split_paragraphs: bool=False, fix_lengths: bool=False, level: str="char",
              **kwargs) -> (LanguageDataset, LanguageDataset):

    # load training and testing data
    if fix_lengths:
        fixed_length = max_chars
    else:
        fixed_length = None

    fields = get_data_fields(fixed_length)
    _paragraph = fields["paragraph"][-1]
    _language = fields["language"][-1]
    _characters = fields['characters'][-1]

    training_data = LanguageDataset(training_text, training_labels, taiga_train_path, fields, split_paragraphs, True, max_chars, level)
    validation_data = LanguageDataset(validation_text, validation_labels, taiga_valid_path, fields, False, False, max_chars_test, level)
    if max_chars_test == -1: max_chars_test = max_chars
    testing_data = LanguageDataset(testing_text, testing_labels, taiga_test_path, fields, False, False, max_chars_test, level)


    _paragraph.build_vocab(training_data, min_freq=10) 
    _language.build_vocab(training_data)
    _characters.build_vocab(training_data, min_freq=10)

    return training_data, validation_data, testing_data