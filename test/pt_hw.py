print("\n\n\n ##### PyTorch: Version and Device check #####\n\n")

import torch
print("*** PyTorch version      : ", torch.__version__)

import torchaudio
print("   *** PyTorch Audio     : ", torchaudio.__version__)

import torchvision
print("   *** PyTorch Vision    : ", torchvision.__version__)

print("")
print("(!! the following is build device specific, and here only to confirm hardware availability, ignore !!)")

use_cuda = torch.cuda.is_available()
if use_cuda:
    cdc = torch.cuda.device_count()
    print("*** GPU(s) available:", cdc)
    print('    CUDNN:', torch.backends.cudnn.version())

    c = 0
    while c < cdc:
        print('    CUDA {} Device Name/ Memory (GB): {} / {}'.format(c, torch.cuda.get_device_name(c), torch.cuda.get_device_properties(0).total_memory/1e9))
        c += 1
else:
    print("*** CPU only")

# From the above, select first device
# device = torch.device("cuda:0" if use_cuda else "cpu")
# print("Device: ",device)
