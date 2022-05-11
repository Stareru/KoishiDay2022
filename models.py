import json, re, random
from transformers import GPT2LMHeadModel, AutoTokenizer

import numpy as np
import torch
from torch import nn

from tqdm.notebook import tqdm


class TouhouMusicTranslator:

    def __init__(self, state_dict, device='cuda'):

        self.plm_jp = 'rinna/japanese-gpt2-medium'
        self.plm_zh = 'junnyu/wobert_chinese_plus_base'
        self.tok_jp = AutoTokenizer.from_pretrained(self.plm_jp)
        self.tok_zh = AutoTokenizer.from_pretrained(self.plm_zh)
        self.model = GPT2LMHeadModel.from_pretrained(self.plm_jp)

        self.tokens_zh = list(self.tok_zh.get_vocab().keys())
        self.tok_jp.add_tokens(sorted(self.tokens_zh))
        self.model.resize_token_embeddings(len(self.tok_jp))

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(state_dict, map_location=self.device))

        self.vocabs = self.tok_jp.get_vocab()
        self.vocabs = {self.vocabs[key]: key for key in self.vocabs.keys()}
        self.vocabs = [self.vocabs[idx] for idx, word in enumerate(self.vocabs)]

        outs = [self.tok_jp.unk_token, self.tok_zh.unk_token, self.tok_jp.sep_token, self.tok_jp.bos_token]
        keys_zh = list(self.tok_zh.get_vocab().keys())
        outs.extend(list(set(self.vocabs).difference(set(keys_zh))))
        outs.remove(self.tok_jp.eos_token)
        self.mask = torch.BoolTensor([vocab in outs for vocab in self.vocabs]).to(self.device)

    def translate(self, ja, beam_size=6):

        tok_jp = self.tok_jp
        tok_zh = self.tok_zh
        model = self.model
        vocabs = self.vocabs

        zh = ''

        tokens = tok_jp.tokenize(ja) + [tok_jp.sep_token] + tok_zh.tokenize(zh)

        res = None
        ids = [tok_jp.convert_tokens_to_ids(tokens)]
        ids = torch.LongTensor(ids).to(self.device)

        probs_cs = torch.FloatTensor([1.0]).to(self.device)

        with torch.no_grad():
            for step in range(64):

                logits = model(ids)['logits'][:, -1]
                logits.masked_fill_(self.mask, -1e8)
                probs = logits.softmax(-1)

                probs_c, ids_c = torch.sort(probs, -1, descending=True)

                probs_c, ids_c = probs_c[:, :beam_size], ids_c[:, :beam_size]
                probs_cs = (probs_c * probs_cs.unsqueeze(-1)).reshape(-1)

                ids = torch.cat((ids.unsqueeze(1).repeat(1, beam_size, 1), ids_c.unsqueeze(-1)), -1)
                ids = ids.reshape(-1, ids.shape[-1])

                probs_cs, rank = torch.sort(probs_cs, -1, descending=True)
                probs_cs, rank = probs_cs[:beam_size], rank[:beam_size]

                ids = ids[rank]

                if ids[0, -1].item() == tok_jp.eos_token_id:
                    break

            tokens = tok_jp.convert_ids_to_tokens(ids[0, :-1].detach().cpu().numpy())
            res = tok_jp.convert_tokens_to_string(tokens).replace(tok_jp.sep_token, '\n')

            return res.split('\n')[1].replace('##', '').split('</s>')[0] if res is not None else ''


class TouhouMusicGenerator:

    def __init__(self, state_dict, device='cuda', translator=None):
        self.plm = 'rinna/japanese-gpt2-medium'
        self.tok = AutoTokenizer.from_pretrained(self.plm)
        self.device = torch.device(device)
        self.model = GPT2LMHeadModel.from_pretrained(self.plm).to(self.device)
        self.model.load_state_dict(torch.load(state_dict, map_location=self.device))
        self.vocabs = self.tok.get_vocab()
        self.vocabs = {self.vocabs[key]: key for key in self.vocabs.keys()}
        self.vocabs = [self.vocabs[idx] for idx, word in enumerate(self.vocabs)]
        self.translator = translator

    def generate(self, prefix, temperature=1.0, step_range=(256, 512), log=False):

        tok = self.tok
        model = self.model
        vocabs = self.vocabs

        step_min, step_max = step_range

        tokens = tok.tokenize(tok.bos_token + prefix)

        lyrics, lyric = [], tokens[1:]

        with torch.no_grad():
            for step in range(step_max):
                ids = tok.convert_tokens_to_ids(tokens)
                ids = torch.LongTensor(ids).to(self.device)

                if step < step_min:
                    outs = [tok.unk_token, tok.eos_token, tok.bos_token]
                else:
                    outs = [tok.unk_token, tok.bos_token]

                mask = torch.BoolTensor([vocab in outs for vocab in vocabs]).to(self.device)

                logits = model(ids)['logits'][-1]
                logits = logits.masked_fill_(mask, -1e8)
                probs = (logits / temperature).softmax(0)
                probs = probs.detach().cpu().numpy()

                token_next = np.random.choice(vocabs, p=probs)
                if token_next == tok.eos_token:
                    break
                elif token_next == tok.sep_token:
                    lyric = tok.convert_tokens_to_string(lyric)
                    if self.translator is not None:
                        translation = self.translator.translate(lyric)
                        lyric = {
                            'lyric': lyric,
                            'translation': translation,
                        }
                        if log:
                            print('歌词：', lyric['lyric'])
                            if 'translation' in lyric.keys():
                                print('翻译：', lyric['translation'])
                            print('-' * 50)
                    else:
                        lyric = {
                            'lyric': lyric,
                        }
                    lyrics.append(lyric)
                    lyric = []
                else:
                    lyric.append(token_next)

                tokens.append(token_next)

        return lyrics
    
    def load(self, state_dict):
        
        self.model.load_state_dict(torch.load(state_dict, map_location=self.device))