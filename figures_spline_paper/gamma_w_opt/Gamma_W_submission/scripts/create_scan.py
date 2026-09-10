import numpy as np
import matplotlib.pyplot as plt
import os

class RippleInducer:
    def __init__(self, vmec_input):
        with open(vmec_input, 'r') as f:
            self.lines = f.readlines()
        for l in self.lines:
            if 'RBC(' in l:               
                self.r00 = float(l.split()[4][:-1])
                break
        else:
            raise ValueError('RBC(0,0) not found!! NOTE: the logic to find this is very stupid and fragile and does not generalize to any VMEC file. Check the number of spaces in the "RBC(   0,   0) =  1.367375147766000e+01," line in the file.')
        self.original_lines = self.lines[:] # copy

    def add_ripple(self, n,m,amplitude, z=False, relative=True):
        search_string = 'RBC'

        if relative:
            scale_factor = self.r00
        else:
            scale_factor = 1.0
            
        if z:
            isplit = 1
            split_str = "ZBS"
        else:
            isplit = 0
            split_str = "ZBS"
        
        for i,l in enumerate(self.lines):
            if search_string in l:
                tmp = l.split(split_str,1)
                tmp2 = tmp[isplit]
                tmp3 = tmp2.split()
                _n = int(tmp3[1][:-1])
                _m = int(tmp3[2][:-1])
                # P5a fix: strip ONLY a trailing comma, not the last character. The old
                # `[:-1]` ate the last char, which for the perturbed/last coeff on a line
                # (no trailing comma) was the sci-notation exponent digit: e-01 -> e-0 (x10).
                _amplitude = float(tmp3[4].rstrip(','))

                if _n == n and _m == m:

                    if z:
                        prefix = tmp[0]  + split_str
                        postfix = '\n'
                    else:
                        postfix = ",    "  + split_str + tmp[1]
                        prefix = 'RBC'
                    
                    A =  _amplitude  + amplitude * scale_factor
                    new_line = prefix + "(   " + str(n) +",   "+str(m)+") =  " + str(A) + postfix
                    self.lines[i] = new_line


    def reset(self):
        # P5b fix: restore a fresh COPY. The old `self.lines = self.original_lines` aliased
        # the two lists, so the next add_ripple's in-place `self.lines[i] = ...` mutated
        # original_lines too -> ripple amplitudes ACCUMULATED across scan iterations.
        self.lines = self.original_lines[:]
        
    def write_new_input(self, new_filename):
        with open(new_filename, 'w') as f:
            f.write("".join(self.lines))


def fl2str(fl, n):
    return ("{:." + str(n)+ "f}").format(fl).replace('-','m').replace('.','p')

            
def create_scan(og_input, n,m, amplitudes, z=False):   
    def dirname(a,n,m,z):
        if z:
            prefix = 'Z_'
        else:
            prefix = ''
        return prefix + str(n) + "_" + str(m) + "/" +  fl2str(a, 5)
        
    ri = RippleInducer(og_input)
    for a in amplitudes:
        d = dirname(a,n,m,z)
        if not os.path.exists(d):
            os.makedirs(d)
            ri.add_ripple(n, m , a, z=z)
            ri.write_new_input(d + "/input.vmec")
            ri.reset()
            
        
if __name__ == "__main__":
    ri = RippleInducer('original/input.vmec')
    #ri.add_ripple(1 , 0 , 0.3)
    ri.add_ripple(1 , 0 , 0.3, z=True)

    amplitudes = np.concatenate((np.array([0.0]),np.logspace(-2,np.log10(0.2),10)))
    create_scan('original/input.vmec',2,0,amplitudes)
    
