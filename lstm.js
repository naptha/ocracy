/*
 * This is a port of Thomas M. Breuel's lstm.py from OCROpus to Javascript
 * It's meant to be compatible with parameters trained with OCROpus' pyrnn
 * and clstm, so I've only implemented the methods for forward propagation.
 *
 * Authors: Kevin Kwok <antimatter15@gmail.com>
 *          Thomas M. Breuel (OCROpus and clstm)
 *
 * License: MIT License
 */

// Nonlinearity used for gates.
function ffunc(vec){
    var arr = new Float32Array(vec.length);
    for(var i = 0; i < vec.length; i++)
        arr[i] = 1.0 / (1.0 + Math.exp(-vec[i]));
    return arr;
}

// Nonlinearity used for input to state.
function gfunc(vec){
    var arr = new Float32Array(vec.length);
    for(var i = 0; i < vec.length; i++) arr[i] = tanh(vec[i]);
    return arr;
}
// Nonlinearity used for output
function hfunc(vec){
    var arr = new Float32Array(vec.length);
    for(var i = 0; i < vec.length; i++) arr[i] = tanh(vec[i]);
    return arr;
}

// This is a standard LSTM network with all the forward propagation formulas
// optimized for speed. The lack of a real numpy-type standard numerics
// package pretty much means I have to unroll the loops anyway so whatever. 

function LSTM(Ni, Ns) {
    this.Ni = Ni;
    this.Ns = Ns;
    this.Na = Ni + Ns + 1;
    this.allocate(5000)
    this.reset()
}

// Allocate space for the internal state variables, maxlen represents the
// maximum length (width of any line) that can be processed

LSTM.prototype.allocate = function(maxlen){
    this.cix    = mat(maxlen, this.Ns) 
    this.ci     = mat(maxlen, this.Ns) 
    this.gix    = mat(maxlen, this.Ns) 
    this.gi     = mat(maxlen, this.Ns) 
    this.gox    = mat(maxlen, this.Ns) 
    this.go     = mat(maxlen, this.Ns) 
    this.gfx    = mat(maxlen, this.Ns) 
    this.gf     = mat(maxlen, this.Ns)
    this.state  = mat(maxlen, this.Ns)
    this.output = mat(maxlen, this.Ns)
    this.source = mat(maxlen, this.Na)
}

// Reset the internal state variables to NaN (anananananana batman!)
LSTM.prototype.reset = function(){
    rmat(this.cix); rmat(this.ci)
    rmat(this.gix); rmat(this.gi)
    rmat(this.gox); rmat(this.go)
    rmat(this.gfx); rmat(this.gf)
    rmat(this.source)
    rmat(this.state)
    rmat(this.output)
}

// Performs forward propagation of activations and updates internal
// state. The input is a 2D array with rows representing input 
// vectors at each time step. Returns a 2D array whose rows represent
// output vectors for each input vector

LSTM.prototype.forward = function(xs){
    // assert(xs[0].length == this.Ni)
    var n = xs.length,
        N = this.gi.length;
    if(n > N) throw "input too large for LSTM model";
    this.reset();
    var prev = new Float32Array(this.Ns)
    for(var t = 0; t < n; t++){
        // assert(xs[t].length == this.Ni)
        this.source[t][0] = 1
        this.source[t].set(xs[t], 1)
        this.source[t].set(prev, 1 + this.Ni)
        for(var j = 0; j < this.Ns; j++){
            this.gix[t][j] = 0; 
            this.gfx[t][j] = 0; 
            this.gox[t][j] = 0; 
            this.cix[t][j] = 0; 
            for(var k = 0; k < this.Na; k++){
                var sk = this.source[t][k];
                this.gix[t][j] += this.WGI[j][k] * sk
                this.gfx[t][j] += this.WGF[j][k] * sk
                this.gox[t][j] += this.WGO[j][k] * sk
                this.cix[t][j] += this.WCI[j][k] * sk
            }
        }
        if(t > 0){
            addmulvec(this.gix[t], this.WIP, this.state[t-1])
            addmulvec(this.gfx[t], this.WFP, this.state[t-1])
        }
        this.gi[t] = ffunc(this.gix[t])
        this.gf[t] = ffunc(this.gfx[t])
        this.ci[t] = gfunc(this.cix[t])
        setmulvec(this.state[t], this.ci[t], this.gi[t]);
        if(t > 0){
            addmulvec(this.state[t], this.gf[t], this.state[t-1])
            addmulvec(this.gox[t], this.WOP, this.state[t])
        }
        this.go[t] = ffunc(this.gox[t])
        setmulvec(this.output[t], hfunc(this.state[t]), this.go[t])
        prev = this.output[t];
    }
    // nanfree(this.output.slice(0, n))
    return this.output.slice(0, n)
}

// Stack two or more networks on top of each other
function Stacked(nets){ this.nets = nets }

Stacked.prototype.forward = function(xs) {
    for(var i = 0; i < this.nets.length; i++)
        xs = this.nets[i].forward(xs);
    return xs;
}

// A time-reversed network
function Reversed(net){ this.net = net }
Reversed.prototype.forward = function(xs) {
    return this.net.forward(xs.slice(0).reverse()).slice(0).reverse();
};

// Run multiple networks on the same input in parallel
function Parallel(nets){ this.nets = nets }
Parallel.prototype.forward = function(xs) {
    var a = this.nets[0].forward(xs),
        b = this.nets[1].forward(xs);
    var m = mat(a.length, a[0].length + b[0].length)
    for(var i = 0; i < a.length; i++){
        m[i].set(a[i])
        m[i].set(b[i], a[0].length)
    }
    return m;
}

// A softmax layer
function Softmax(Nh, No){
    this.Nh = Nh;
    this.No = No;
}

Softmax.prototype.forward = function(ys){
    var n  = ys.length,
        zs = [];
    for(var i = 0; i < n; i++){
        var temp = new Float32Array(this.No),
            total = 0;
        for(var j = 0; j < this.No; j++){
            var v = this.B[j]
            for(var k = 0; k < this.Nh; k++){
                v += this.W[j][k] * ys[i][k]
            }
            temp[j] = Math.exp(Math.min(100, Math.max(-100, v)))
            total += temp[j]
        }
        for(var j = 0; j < this.No; j++) temp[j] /= total;
        zs[i] = temp;
    }
    return zs;
}

// A bidirectional LSTM constructed from regular and reversed LSTMs.
function BiDiLSTM(Ni, Ns, No){
    var lstm1 = new LSTM(Ni, Ns),
        lstm2 = new Reversed(new LSTM(Ni, Ns)),
        bidi  = new Parallel([lstm1, lstm2]),
        soft  = new Softmax(2 * Ns, No),
        stack = new Stacked([bidi, soft]);
    return stack;
}

function SequenceRecognizer(ninput, nstates, codec) {
    this.codec = codec;
    this.Ni   = ninput;
    this.Ns   = nstates;
    this.No   = codec.length;
    this.lstm = BiDiLSTM(this.Ni, this.Ns, this.No);
}

// predict an integer sequence of codes
SequenceRecognizer.prototype.predictSequence = function(xs) {
    if(xs[0].length != this.Ni) throw "wrong image height";
    this.output = this.lstm.forward(xs);

    // return array_split(this.output
    //     .map(x => Math.max.apply(Math, x) > 0.6 && x)
    //     .filter(x => x)
    //     .map(x => max_index(x))).map(x=>x[0])

    // return array_split(this.output
    //     .map(x => Math.max.apply(Math, x) > 0.6 && x)
    //     .filter(x => x)
    //     .map(x => max_index(x)),
    //     (a,b) => (a == 0 && b == 0) || (a != 0 && b != 0) 
    //     ).map(x=>x[0])

    return array_split(
            this.output.map(x => [max_index(x), Math.max.apply(Math, x)]), 
            (a,b) => (a[0] == 0 && b[0] == 0) || (a[0] != 0 && b[0] != 0)
        )
        .map(k => max_element(k, x => x[1]))
        .filter(k => k[1] > 0.7)
        .map(x => x[0])
}

SequenceRecognizer.prototype.predictString = function(xs) {
    return this.predictSequence(xs).map(x=>this.codec[x]).join('')
}
