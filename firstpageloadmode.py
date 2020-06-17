import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import grpah
import os
from tkinter import *
from PIL import ImageTk,Image
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch import optim
from torch.nn import functional
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def openfile(self):
    print('click')


root = tk.Tk()
root.minsize(640,400)
root.title('Generation')
tk.Label(root, text = 'Live Generation of Synthetic Data',
      font =('Verdana', 20)).pack(pady = 20)

### image

fullpath = os.path.join(os.path.dirname(__file__), 'products2.jpg')

canvas = Canvas(root, width = 640, height = 400)
canvas.pack()
img = ImageTk.PhotoImage(Image.open(fullpath))
canvas.create_image(320, 200, anchor=CENTER, image=img)


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)


with open(os.path.join(os.path.dirname(__file__), 'parameters.pickle'), 'rb') as handle:
        parameter_dict = pickle.load(handle)

def OpenResults(event=None):
    ## MODEL
    # load parameters
    fullpath1 = os.path.join(os.path.dirname(__file__))
    n=e1.get()
    n = int(n)



    n_col = parameter_dict['n_col']
    n_opt = parameter_dict['n_opt']
    model = parameter_dict['model']
    interval = parameter_dict['interval']
    batch_size = parameter_dict['batch_size']
    embedding_dim = parameter_dict['embedding_dim']
    output_info = parameter_dict['output_info']
    meta = parameter_dict['meta']
    n_clusters = parameter_dict['n_clusters']
    delta = parameter_dict['delta']
    unit_columns = parameter_dict['unit_columns']
    dollar_columns = parameter_dict['dollar_columns']
    discrete_columns = parameter_dict['discrete_columns']



    def _apply_activate(data, output_info):
        data_t = []
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    #
    def cond_sample_zero(batch, n_col=n_col, n_opt=n_opt, model=model, interval=interval):
        if n_col == 0:
            return None

        vec = np.zeros((batch, n_opt), dtype='float32')
        idx = np.random.choice(np.arange(n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(model[col]))
            vec[i, pick + interval[col, 0]] = 1

        return vec

    #

    def _inverse_transform_continuous(meta, data, sigma, n_clusters=n_clusters):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(data, sigmas, meta=meta):
        start = 0
        output = []
        column_names = []
        for meta in meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = _inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = _inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names)

        return output

    generator = torch.load(os.path.join(fullpath1,'generator.pth'))
    generator.eval()


    steps = n // batch_size + 1
    data = []
    for i in range(steps):
        mean = torch.zeros(batch_size, embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(torch.device('cpu'))

        condvec = cond_sample_zero(batch_size)

        if condvec is None:
            pass
        else:
            c1 = condvec
            c1 = torch.from_numpy(c1).to(torch.device('cpu'))
            fakez = torch.cat([fakez, c1], dim=1)

        fake = generator(fakez)
        fakeact = _apply_activate(fake, output_info)
        data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]

    # inverse transform the generated data
    generated_data = inverse_transform(data, None)

    generated_data[unit_columns] = generated_data[unit_columns].astype(float)
    generated_data[dollar_columns] = generated_data[dollar_columns].astype(float)
    generated_data[discrete_columns] = generated_data[discrete_columns].astype('category')

    generated_data[unit_columns] = (np.exp(generated_data[unit_columns]) - delta).astype(int)
    generated_data[dollar_columns] = (np.exp(generated_data[dollar_columns]) - delta).clip(0)
    grpah.main(generated_data)


## create button

# and lay them out
top = Frame(root)
top.pack(side=TOP)


#btn2 = tk.Button(top, text='Select Model', command=UploadActionModel, font =('Verdana', 10))
#btn2.pack(pady = 20, side = LEFT, padx=7)
tk.Label(top, text="Number of data points:").grid(row=1)
e1 = tk.Entry(top)
e1.grid(row=1, column=1)




tk.Label(root, text = 'Generate and validate synthetic data',
      font =('Verdana', 12)).pack(pady = 5)

btn3 = tk.Button(root, text='See results', command=OpenResults, font =('Verdana', 15))
btn3.pack(pady = 30)
tk.Label(root, text = 'Copyright © 2020: Carrie Lu, Daniela Matinho, Hannah Kerr, Yuling Gu',
      font =('Verdana', 9)).pack(pady = 5, side=LEFT)

root.mainloop()