# An implementation of LSTM networks, CTC alignment, and related classes.
#
# This code operates on sequences of vectors as inputs, and either outputs
# sequences of vectors, or symbol sequences. Sequences of vectors are
# represented as 2D arrays, with rows representing vectors at different
# time steps.
#
# The code makes liberal use of array programming, including slicing,
# both for speed and for simplicity. All arrays are actual narrays (not matrices),
# so `*` means element-wise multiplication. If you're not familiar with array
# programming style, the numerical code may be hard to follow. If you're familiar with
# MATLAB, here is a side-by-side comparison: http://wiki.scipy.org/NumPy_for_Matlab_Users
#
# Implementations follow the mathematical formulas for forward and backward
# propagation closely; these are not documented in the code, but you can find
# them in the original publications or the slides for the LSTM tutorial
# at http://lstm.iupr.com/
#
# You can find a simple example of how to use this code in this worksheet:
# https://docs.google.com/a/iupr.com/file/d/0B2VUW2Zx_hNoXzJQemFhOXlLN0U
# More complex usage is illustrated by the ocropus-rpred and ocropus-rtrain
# command line programs.
#
# Author: Thomas M. Breuel
# License: Apache 2.0

from pylab import rand, zeros, nan, ones, amax, array, dot, exp, tanh, isnan, concatenate, clip, argmax
import unicodedata


class Codec:
    """Translate between integer codes and characters."""
    def init(self,charset):
        charset = sorted(list(set(charset)))
        self.code2char = {}
        self.char2code = {}
        for code,char in enumerate(charset):
            self.code2char[code] = char
            self.char2code[char] = code
        return self
    def size(self):
        """The total number of codes (use this for the number of output
        classes when training a classifier."""
        return len(list(self.code2char.keys()))
    def encode(self,s):
        "Encode the string `s` into a code sequence."
        # tab = self.char2code
        dflt = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]
    def decode(self,l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s

ascii_labels = [""," ","~"] + [unichr(x) for x in range(33,126)]


def normalize_nfkc(s):
    return unicodedata.normalize('NFKC',s)

def ascii_codec():
    "Create a codec containing just ASCII characters."
    return Codec().init(ascii_labels)

def ocropus_codec():
    """Create a codec containing ASCII characters plus the default
    character set from ocrolib."""
    import ocrolib
    base = [c for c in ascii_labels]
    base_set = set(base)
    extra = [c for c in ocrolib.chars.default if c not in base_set]
    return Codec().init(base+extra)

def getstates_for_display(net):
    """Get internal states of an LSTM network for making nice state
    plots. This only works on a few types of LSTM."""
    if isinstance(net,LSTM):
        return net.state[:net.last_n]
    if isinstance(net,Stacked) and isinstance(net.nets[0],LSTM):
        return net.nets[0].state[:net.nets[0].last_n]
    return None




initial_range = 0.1

class RangeError(Exception):
    def __init__(self,s=None):
        Exception.__init__(self,s)

def randu(*shape):
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution for weight initializations.
    Empirically, the choice of randu/randn can make a difference
    for neural network initialization."""
    return 2*rand(*shape)-1

def sigmoid(x):
    """Compute the sigmoid function.
    We don't bother with clipping the input value because IEEE floating
    point behaves reasonably with this function even for infinities."""
    return 1.0/(1.0+exp(-x))



class Softmax():
    """A softmax layer, a straightforward implementation
    of the softmax equations. Uses 1-augmented vectors."""
    
    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh = Nh
        self.No = No
        self.W2 = zeros((No,Nh+1))*initial_range
    
    def forward(self, ys):
        """Forward propagate activations. This updates the internal
        state for a subsequent call to `backward` and returns the output
        activations."""
        n = len(ys)
        # inputs, zs = [None]*n,[None]*n
        zs = [None] * n

        for i in range(n):
            # inputs[i] = concatenate([ones(1), ys[i]])
            # print self.W2[:,0]

            temp = dot(self.W2[:,1:], ys[i]) + self.W2[:,0]
            # print 'yss', ys[i].shape, self.W2[:,1:].shape, temp.shape
            # temp = dot(self.W2, inputs[i])
            # print temp - dot(self.W2[:,1:], ys[i]) - self.W2[:,0]
            # print inputs[i].shape, self.W2.shape, temp.shape
            # print self.W2[i], i, n
            # temp = dot(self.W2[:,1:], ys[i]) + self.W2[:,0]
            temp = exp(clip(temp,-100,100))
            temp /= sum(temp)
            zs[i] = temp
        # self.state = (inputs,zs)
        return zs

# These are the nonlinearities used by the LSTM network.
# We don't bother parameterizing them here

def ffunc(x):
    "Nonlinearity used for gates."
    return 1.0/(1.0+exp(-x))

def gfunc(x):
    "Nonlinearity used for input to state."
    return tanh(x)

# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return tanh(x)


class LSTM():
    """A standard LSTM network. This is a direct implementation of all the forward
    and backward propagation formulas, mainly for speed. (There is another, more
    abstract implementation as well, but that's significantly slower in Python
    due to function call overhead.)"""
    def __init__(self,ni,ns,initial=initial_range,maxlen=5000):
        na = 1+ni+ns
        self.dims = ni,ns,na
        self.init_weights(initial)
        self.allocate(maxlen)
    
    def init_weights(self,initial):
        "Initialize the weight matrices and derivatives"
        ni,ns,na = self.dims
        # gate weights
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,zeros((ns,na)))
        # peep weights
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,zeros(ns))
    
    def allocate(self,n):
        """Allocate space for the internal state variables.
        `n` is the maximum sequence length that can be processed."""
        ni,ns,na = self.dims
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output"
        for v in vars.split():
            setattr(self,v,nan*ones((n,ns)))
        self.source = nan*ones((n,na))
        # self.sourceerr = nan*ones((n,na))
    
    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " source state output"

        for v in vars.split():
            getattr(self,v)[:,:] = nan

    def forward(self,xs):
        """Perform forward propagation of activations and update the
        internal state for a subsequent call to `backward`.
        Since this performs sequence classification, `xs` is a 2D
        array, with rows representing input vectors at each time step.
        Returns a 2D array whose rows represent output vectors for
        each input vector."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        # self.last_n = n
        N = len(self.gi)
        if n>N: raise ocrolib.RecognitionError("input too large for LSTM model")
        self.reset(n)

        # Both functions are a straightforward implementation of the
        # LSTM equations. It is possible to abstract this further and
        # represent gates and memory cells as individual data structures.
        # However, that is several times slower and the extra abstraction
        # isn't actually all that useful.

        """Perform forward propagation of activations for a simple LSTM layer."""
        for t in range(n):
            prev = zeros(ns) if t == 0 else self.output[t-1]
            self.source[t,0]      = 1
            self.source[t,1:1+ni] = xs[t]
            self.source[t,1+ni:]  = prev
            self.gix[t] = dot(self.WGI,self.source[t])
            self.gfx[t] = dot(self.WGF,self.source[t])
            self.gox[t] = dot(self.WGO,self.source[t])
            self.cix[t] = dot(self.WCI,self.source[t])
            if t > 0:
                self.gix[t] += self.WIP * self.state[t-1]
                self.gfx[t] += self.WFP * self.state[t-1]
            self.gi[t] = ffunc(self.gix[t])
            self.gf[t] = ffunc(self.gfx[t])
            self.ci[t] = gfunc(self.cix[t])
            self.state[t] = self.ci[t] * self.gi[t]
            if t > 0:
                self.state[t] += self.gf[t] * self.state[t-1]
                self.gox[t]   += self.WOP   * self.state[t]
            self.go[t]     = ffunc(self.gox[t])
            self.output[t] = hfunc(self.state[t]) * self.go[t]
        
        assert not isnan(self.output[:n]).any()
        return self.output[:n]

################################################################
# combination classifiers
################################################################


class Stacked():
    """Stack two networks on top of each other."""
    def __init__(self,nets):
        self.nets = nets
    
    def forward(self,xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs

class Reversed():
    """Run a network on the time-reversed input."""
    def __init__(self,net):
        self.net = net
    
    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]

class Parallel():
    """Run multiple networks in parallel on the same input."""
    def __init__(self,*nets):
        self.nets = nets
    
    def forward(self,xs):
        # print 'xs shape', xs.shape
        outputs = [net.forward(xs) for net in self.nets]
        # print 'out1', len(outputs)
        outputs = zip(*outputs)
        # print 'out2', len(outputs)
        # print 'out2', len(outputs), [(x.shape, y.shape) for x,y in outputs]
        outputs = [concatenate(l) for l in outputs]
        # print 'out3', len(outputs), [x.shape for x in outputs]
        # print outputs
        return outputs

def BIDILSTM(Ni,Ns,No):
    """A bidirectional LSTM, constructed from regular and reversed LSTMs."""
    assert No>1
    lstm1 = LSTM(Ni, Ns)
    lstm2 = Reversed(LSTM(Ni, Ns))
    bidi = Parallel(lstm1, lstm2)
    logreg = Softmax(2*Ns, No)
    stacked = Stacked([bidi, logreg])
    return stacked

def translate_back0(outputs,threshold=0.7):
    """Simple code for translating output from a classifier
    back into a list of classes. TODO/ATTENTION: this can
    probably be improved."""
    ms = amax(outputs,axis=1)
    cs = argmax(outputs,axis=1)
    cs[ms<threshold] = 0
    result = []
    for i in range(1,len(cs)):
        if cs[i]!=cs[i-1]:
            if cs[i]!=0:
                result.append(cs[i])
    return result


from scipy.ndimage import measurements,filters
from pylab import tile, arange

def translate_back(outputs,threshold=0.7,pos=0):
    """Translate back. Thresholds on class 0, then assigns
    the maximum class to each region."""
    # print outputs
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = tile(labels.reshape(-1,1), (1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,arange(1,amax(mask)+1))
    if pos: return maxima
    return [c for (r,c) in maxima]

class SeqRecognizer:
    """Perform sequence recognition using BIDILSTM and alignment."""
    def __init__(self,ninput,nstates,noutput=-1,codec=None,normalize=None):
        self.Ni = ninput
        if codec: noutput = codec.size()
        print "noutput", noutput
        assert noutput>0
        self.No = noutput
        self.lstm = BIDILSTM(ninput,nstates,noutput)
        # self.normalize = normalize
        self.codec = codec
    
    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert xs.shape[1]==self.Ni,\
            "wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni)
        self.outputs = array(self.lstm.forward(xs))
        return translate_back(self.outputs)

    def l2s(self,l):
        "Convert a code sequence into a unicode string after recognition."
        l = self.codec.decode(l)
        return u"".join(l)
    
    def predictString(self,xs):
        "Predict output as a string. This uses codec and normalizer."
        cs = self.predictSequence(xs)
        return self.l2s(cs)
