import matplotlib.pyplot as plt
def plot_acc(depths, train_accs, test_accs, val_accs=[]):
    plt.plot(depths, train_accs, label='Train F1-Score', marker='o')
    # plt.plot(depths, val_accs, label='Val Accuracy', marker='o')
    plt.plot(depths, test_accs, label='Test F1-Score', marker='o')
    plt.xlabel('Number of Hidden Layer')
    plt.ylabel('Average F1-Score')
    plt.legend()
    plt.grid()
    plt.show()
    
depths = ["[512]", "[512,256]", "[512,256,128]", "[512,256,128,64]"]
train_accs = [0.9060,0.9201,0.9098,0.8955]
test_accs = [0.7721,0.7930,0.7754,0.7644]
plot_acc(depths, train_accs,test_accs)
