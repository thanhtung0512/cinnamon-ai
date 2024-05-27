import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import Resource, Api
import gc
import cv2
import sys
import math
import time
import timm
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)
class Timm(nn.Module):
    def __init__(self, name, hidden = 256, drop_path_rate=0.1, drop_rate=0.1, dropout=0.5):
        super(Timm, self).__init__()

        base_model = timm.create_model(name, pretrained=False, 
            drop_rate = drop_rate,
            drop_path_rate =drop_path_rate
            )

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(in_features, hidden, 1)

    def forward(self, x):
        """
        Shape: 
            - x: (N, C, H, W)
            - output: (W, N, C)
        """
        conv = self.encoder(x)

        # conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        # print(conv.shape) #BxChx4x64

        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)

        conv = conv.permute(-1, 0, 1)


        return conv


class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
                
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        # self.rnn1 = nn.GRU(enc_hid_dim*2, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x hid_dim 
        hidden: batch_size x hid_dim
        """

        embedded = self.dropout(src)
        
        outputs, hidden = self.rnn(embedded)
        # outputs, hidden = self.rnn1(outputs)
                                 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim,
        outputs: batch_size x src_len
        """
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        # self.rnn1 = nn.GRU(dec_hid_dim * 2, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """
             
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
                
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output, hidden = self.rnn1(output, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, img_channel, decoder_embedded, dropout=0.1):
        super().__init__()
        
        attn = Attention(encoder_hidden, decoder_hidden)
        
        self.encoder = Encoder(img_channel, encoder_hidden, decoder_hidden, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, encoder_hidden, decoder_hidden, dropout, attn)
        
    def forward_encoder(self, src):       
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size 
        hidden: batch_size x hid_dim
        encouder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """
        
        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)
        
        return output, (hidden, encoder_outputs)

    def forward(self, src, trg):
        """
        src: time_step x batch_size
        trg: time_step x batch_size
        outputs: batch_size x time_step x vocab_size
        """

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
                
        ##TODO reverse the order>> decode backward??
        for t in range(trg_len):
            input = trg[t] 
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            outputs[t] = output
            
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs

    def expand_memory(self, memory, beam_size):
        hidden, encoder_outputs = memory
        hidden = hidden.repeat(beam_size, 1)
        encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

        return (hidden, encoder_outputs)
    
    def get_memory(self, memory, i):
        hidden, encoder_outputs = memory
        hidden = hidden[[i]]
        encoder_outputs = encoder_outputs[:, [i],:]

        return (hidden, encoder_outputs)

class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}
        print(f'c2i: {self.c2i}')
        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars if c in self.c2i] + [self.eos]
    
    def decode(self, ids):
        print(f"-------DECODE-------")
        first = 1 if self.go in ids else 0
        print(f'first: {first}')
        last = ids.index(self.eos) if self.eos in ids else None
        print(f'last: {last}')
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        print(f'sent: {sent}')
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars

class VietOCR(nn.Module):
    def __init__(self, vocab, hidden_dim, backbone):
        
        super(VietOCR, self).__init__()

        self.vocab = vocab
        self.cnn = Timm(backbone, hidden = hidden_dim, drop_path_rate=0.5, drop_rate=0.5, dropout=0.5)

        self.transformer = Seq2Seq(len(self.vocab), encoder_hidden=hidden_dim, decoder_hidden=hidden_dim, img_channel=hidden_dim, decoder_embedded=hidden_dim, dropout=0.5) 

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        # print(src.shape)
        outputs = self.transformer(src, tgt_input)

        return outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab =  'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '
vocab = Vocab(vocab)

vocab1 =  'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~ '

##load model
model_b1 = VietOCR(vocab, hidden_dim = 384, backbone = 'tf_efficientnetv2_b1.in1k')
model_b1.to(device)
checkpoint = torch.load('weight/kalapa_ocr_weights/b1_384_f5_all_swa.pth', map_location="cpu")
model_b1.load_state_dict(checkpoint)
model_b1.eval()

model_b2 = VietOCR(vocab, hidden_dim = 256, backbone = 'tf_efficientnetv2_b2.in1k')
model_b2.to(device)
checkpoint = torch.load('weight/kalapa_ocr_weights/b2_256_f5_all_swa.pth', map_location="cpu")
model_b2.load_state_dict(checkpoint)
model_b2.eval()
print('model loaded!!')
##~load model

def preprocess(image, im_w, im_h, device):
    img_b1 = cv2.resize(image.copy(), (im_w, im_h))
    img_b1 = img_b1.transpose(2,0,1)
    img_b1 = img_b1/255 
    img_b1 = torch.from_numpy(img_b1).unsqueeze(0).float().to(device)
    return img_b1


@app.route('/predict', methods=['POST'])
def predict():
    # Check if image is provided
    if 'image_path' not in request.json:
        return jsonify({'error': 'No image path provided'})

    image_path = request.json['image_path']
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'})
    # Read and decode image
    # image_file = request.files['image']
    # image_np = np.frombuffer(image_file.read(), np.uint8)
    image_cv = cv2.imread(image_path)

    # Start inference timer
    start_time = time.time()
    max_seq_length=128
    sos_token=1
    eos_token=2
    # Preprocess image
    img_b1 = preprocess(image_cv, 1664, 160, device)
    img_b2 = preprocess(image_cv, 2048, 128, device)

    # Run inference
    with torch.no_grad():
        src1 = model_b1.cnn(img_b1)
        src2 = model_b2.cnn(img_b2)

        print(f'src1 shape: {src1.shape}\n src2 shape: {src2.shape}')



        memory1 = model_b1.transformer.forward_encoder(src1)

        memory2 = model_b2.transformer.forward_encoder(src2)

        print(f'Len img_b1 = {len(img_b1)}')
        print(f'img_b1 shape : {img_b1.shape}')
        translated_sentence = [[sos_token]*len(img_b1)]
        
        char_probs = [[1]*len(img_b1)]
        print(f'Translated_sentence: {translated_sentence} \n char_probs: {char_probs}')
        max_length = 0
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
            output1, memory1 = model_b1.transformer.forward_decoder(tgt_inp, memory1)
            output1 = torch.nn.functional.softmax(output1, dim=-1)
            output1 = output1.to('cpu')
            
            output2, memory2 = model_b2.transformer.forward_decoder(tgt_inp, memory2)
            output2 = torch.nn.functional.softmax(output2, dim=-1)
            output2 = output2.to('cpu')
            
            output = 0.5*output1 + 0.5*output2
            
            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            # print(f'Indices = {indices[0]}\n')
            
            values = values[:, -1, 0]
            values = values.tolist()
            # print(f'Value: {values}')
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output
            
        translated_sentence = np.asarray(translated_sentence).T
        print(f'translated_sentence: {translated_sentence[0]}')
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
        
        s = translated_sentence[0].tolist()
        for ind in translated_sentence[0]:
            print(f'{vocab1[ind-4]}')
        answer = vocab.decode(s)
        
    end = time.time()
    

    # End inference timer
    end_time = time.time()
    inference_time = end_time - start_time

    # Return response
    return jsonify({
        'prediction': answer,
        'inference_time': inference_time
    })


# Run the Flask app
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
