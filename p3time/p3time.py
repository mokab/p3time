import torch
import math
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader

class Fourierbasis(torch.nn.Module):
    def __init__(self, n_term: int, 
                 out_features: int, 
                 omega0: float = math.pi,
                 bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        super().__init__()
        self.omega0 = omega0
        self.n_term = n_term
        
        in_features = 2*n_term
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch.nn.parameter.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.phase = torch.nn.parameter.Parameter(torch.empty((1,1), **factory_kwargs))
        torch.nn.init.zeros_(self.phase)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # The number of Fourierbasis
        n_term = self.n_term
        omega0 = self.omega0
        coeff_t = omega0 * torch.linspace(1, n_term, n_term, device = self.device)
        coeff_t = torch.reshape(coeff_t, (1,-1))
        newtt = torch.reshape(input, (-1, 1))
        theta = torch.mm(newtt, coeff_t)
        mat_sin = torch.sin(theta)
        mat_cos = torch.cos(theta)
        MM = torch.cat((mat_sin, mat_cos), 1)

        return torch.nn.functional.linear(MM, self.weight, self.bias)

class Encoder(torch.nn.Module):
    def __init__(self, input_size, layer1_out=30, layer2_out=20, device=None) -> None:
        super().__init__()
        self.encf1 = torch.nn.Linear(input_size, layer1_out, device = device)
        self.encf2 = torch.nn.Linear(layer1_out, layer2_out, device = device)
        self.encf3 = torch.nn.Linear(layer2_out, 1, device = device)
        self.encf4 = torch.nn.Linear(1, 1, bias=False, device = device)
        self.dropout = torch.nn.Dropout(p=0.0, inplace=False)
        # init paramaters
        torch.nn.init.xavier_normal_(self.encf1.weight)
        torch.nn.init.xavier_normal_(self.encf2.weight)
        torch.nn.init.xavier_normal_(self.encf3.weight)
    def forward(self, x):
        x = self.encf1(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.encf2(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.encf3(x)
        x = torch.tanh(x) # x \in (-1,1)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size, n_term, layer1_out=20, layer2_out=30, device=None, dtype=None) -> None:
        super().__init__()
        self.decf1 = torch.nn.Linear(1, layer1_out, bias=False, device = device)
        self.decf2 = torch.nn.Linear(layer1_out, layer2_out, bias=False, device = device)
        self.decf3 = torch.nn.Linear(layer2_out, output_size, bias=False, device = device)
        self.decfb = Fourierbasis(n_term=n_term, out_features=output_size, bias=True, omega0=math.pi, device = device)
        # paramater init 
        torch.nn.init.xavier_normal_(self.decf1.weight)
        torch.nn.init.xavier_normal_(self.decf2.weight)
        torch.nn.init.xavier_normal_(self.decf3.weight)
        torch.nn.init.xavier_normal_(self.decfb.weight)

    def forward(self, tt) -> torch.tensor :
        #---
        X1 = self.decf1(tt)
        X1 = nn.functional.relu(X1)
        X1 = self.decf2(X1)
        X1 = nn.functional.relu(X1)
        X1 = self.decf3(X1)
        #---
        X2 = self.decfb(tt)
        self.X2 = X2
        self.X1 = X1
        Xhat = X1 + X2
        return Xhat

class P3timemodel(torch.nn.Module):
    def __init__(self, size, n_term, middle_width, device = None) -> None:
        super().__init__()
        self.size = size
        self.n_term = n_term
        self.encoder = Encoder(input_size = size, layer1_out=middle_width[0], layer2_out=middle_width[1], device = device)
        self.decoder = Decoder(output_size = size, n_term = n_term, layer1_out=middle_width[1], layer2_out=middle_width[0], device = device)

    def forward(self, XX) -> torch.tensor :
        tt = self.encoder(XX)
        Xhat = self.decoder(tt)
        return Xhat
        
    def predict_pseudotime(self, XX) -> torch.tensor :
        tt = self.encoder(XX).reshape(-1).cpu().detach()
        return tt

class p3time():
    def __init__(self, size, n_term=1, lr = 1e-3, steps = 3000, batch_size = 32, lambdae = 1e-5, lambdad = 1e-6, lambdap = 1e-6, middle_width=[30, 20], device = None, verbose=1000) -> None:
        self.size = size
        self.n_term = n_term
        self.device = device
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.steps = steps
        self.lambdae = lambdae
        self.lambdad = lambdad
        self.lambdap = lambdap
        self.middle_width = middle_width
        self.batch_size = batch_size
        self.verbose = verbose
        self.losslist = []
        self.p3timemodel = P3timemodel(size = self.size, n_term = self.n_term, middle_width = self.middle_width, device = self.device,)
        
    def fit(self, XX) -> None:
        XX = XX.to(torch.float32)
        XX = XX.to(self.device)
        opt = torch.optim.Adam(params=self.p3timemodel.parameters(), lr = self.lr)
        trainloader = DataLoader(XX, batch_size=self.batch_size, shuffle=True)

        for step in range(self.steps):
            for counter, batchX in enumerate(trainloader, 1):
                batchXhat = self.p3timemodel(batchX)
                #regulalized term
                Rp = torch.tensor(0., requires_grad=True)
                Re = torch.tensor(0., requires_grad=True)
                Rd = torch.tensor(0., requires_grad=True)
                # ---
                Rp = Rp + torch.square(self.p3timemodel.state_dict(keep_vars=True)['decoder.decfb.weight']).sum()
                # ---
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf1.weight']).sum()
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf2.weight']).sum()
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf3.weight']).sum()
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf1.bias']).sum()
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf2.bias']).sum()
                Re = Re + torch.square(self.p3timemodel.state_dict(keep_vars=True)['encoder.encf3.bias']).sum()
                # ---
                Rd = Rd + torch.square(self.p3timemodel.state_dict(keep_vars=True)['decoder.decf1.weight']).sum()
                Rd = Rd + torch.square(self.p3timemodel.state_dict(keep_vars=True)['decoder.decf2.weight']).sum()
                Rd = Rd + torch.square(self.p3timemodel.state_dict(keep_vars=True)['decoder.decf3.weight']).sum()

                loss = self.loss(batchXhat , batchX)
                regterm = self.lambdap * Rp + self.lambdae * Re + self.lambdad * Rd
                loss = loss + regterm

                opt.zero_grad()
                loss.backward()
                opt.step()

                self.losslist.append(loss.cpu().detach().numpy())
            if step % self.verbose == 0:
                print("Iter {0}. Loss {1}".format(step, loss))

    def save_model(self, filepath) -> None:
        torch.save(self.p3timemodel.state_dict(), '%s'%filepath)    
        
    def load_model(self, filepath) -> None:
        if torch.cuda.is_available():
            self.p3timemodel.load_state_dict(torch.load('%s'%filepath)) 
        else:
            self.p3timemodel.load_state_dict(torch.load('%s'%filepath, map_location=torch.device('cpu'))) 
            
    def predict_pseudotime(self, XX) -> torch.tensor :
        XX = XX.to(torch.float32)
        XX = XX.to(self.device)
        tt = self.p3timemodel.predict_pseudotime(XX)
        return tt
