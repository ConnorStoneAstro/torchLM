import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

class CS():
    """
    Confidence step optimizer
    """

    def __init__(self, function, lambda0, epsilon4 = 1e-1, max_iter = None, L0 = 1e-3, Lup = 1.11, Ldn = 1.21, Lrej = 1.5, momentum = 0.9, lift = 1.09, lower = 1.5):
        self.function = function
        self.L = L0
        self.Lup = Lup
        self.Ldn = Ldn
        self.Lrej = Lrej
        self.lambda_new = lambda0
        self.max_iter = len(lambda0)*100 if max_iter is None else max_iter
        self.grad_history = []
        self.L_history = []
        self.lambda_history = []
        self.loss_history = []
        self.epsilon4 = epsilon4
        self.momentum = 0.9
        self.lift = lift
        self.lower = lower
        self.main_loop()
        
    def main_loop(self):
        self.iteration = 0
        height = 1e0
        weight = 1 / (1 + height)
        h = torch.zeros(len(self.lambda_new))
        momentum = torch.zeros(len(self.lambda_new))
        while self.iteration < self.max_iter:

            if self.iteration > 0:
                h = -((1 - weight)*grad + weight*momentum) * self.L
            temp_lambda = torch.tensor(self.lambda_new + h, requires_grad = True)
            loss = self.function(temp_lambda)
            
            if self.iteration > 0:
                rho = (min(self.loss_history) - loss.detach().cpu().item()) / (torch.linalg.norm(h).detach().cpu().item())
                if self.iteration > 1:
                    print(loss, rho, np.arccos((torch.dot(grad,torch.tensor(self.grad_history[-2]))/(torch.linalg.norm(grad)*torch.linalg.norm(torch.tensor(self.grad_history[-2])))).detach().item())*180/np.pi)
                    angle = np.arccos((torch.dot(grad,torch.tensor(self.grad_history[-2]))/(torch.linalg.norm(grad)*torch.linalg.norm(torch.tensor(self.grad_history[-2])))).detach().item())*180/np.pi < 45
                else:
                    angle = True
                if rho >= 1. and angle:
                    print("accept over")
                    self.L = min(1e2, self.L*self.Lup)
                    height = min(1e9, height * self.lift)
                    weight = 1 / (1+height)
                    self.lambda_new += h
                elif self.epsilon4 < rho:
                    print("accept under")
                    self.L = max(1e-9, self.L/self.Ldn)
                    height = min(1e9, height / self.lower)
                    weight = 1 / (1+height)
                    self.lambda_new += h
                else:
                    print("reject")
                    self.L = max(1e-9, self.L/self.Lrej)
                    height = min(1e9, height * self.lift)
                    weight = 1 / (1+height)
                    continue

            if self.iteration > 0:
                momentum = self.momentum*(grad + momentum)
            loss.backward()
            grad = temp_lambda.grad
            self.loss_history.append(loss.detach().cpu().item())
            self.L_history.append(self.L)
            self.lambda_history.append(np.copy((self.lambda_new + h).detach().cpu().numpy()))
            self.grad_history.append(np.copy(grad.detach().cpu().numpy()))
            self.iteration += 1
            
if __name__ == "__main__":

    global call_counter
    call_counter = 0
    def y_hat(x, theta):
        global call_counter
        call_counter += 1
        return theta[0] * torch.exp(-x / theta[1]) + theta[2] * x * torch.exp(-x/theta[3]) 

    np.random.seed(10)
    theta_true = torch.tensor([20,10,1,50])
    X = torch.tensor(np.random.uniform(0,100,100), dtype = torch.float32)
    Y = torch.tensor(y_hat(X, theta_true).detach().numpy() + np.random.normal(loc = 0, scale = 0.5, size = len(X)), dtype = torch.float32)

    plt.scatter(X.detach().numpy(), Y.detach().numpy())
    plt.plot(np.linspace(0,100,100), y_hat(torch.linspace(0,100,100), theta_true).detach().numpy())
    plt.show()
    def residual(theta, nocount = False):
        # print("theta: ", theta)
        # plt.scatter(X.detach().numpy(), Y.detach().numpy())
        # plt.plot(np.linspace(0,100,100), y_hat(torch.linspace(0,100,100), theta).detach().numpy())
        # plt.show()
        if not nocount:
            global call_counter
            call_counter += 1
        return torch.sum(((Y - y_hat(X, theta))**2)/(0.5**2)) / (100 - 4 + 1)
    
    x0 = torch.tensor([5.,2.,0.2,10.], dtype = torch.float32)

    res = CS(residual, x0) #LM(y_hat, X, Y, x0, W = torch.ones(len(X))*0.5, max_iter = 100)
    print("call counter: ", call_counter)
    plt.plot(range(len(res.loss_history)), np.log10(np.array(res.loss_history)))
    plt.plot(range(len(res.L_history)), np.log10(np.array(res.L_history)))
    plt.show()
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,0])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,1])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,2])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,3])
    # for t in theta_true.detach().numpy():
    #     plt.axhline(t)
    plt.show()
