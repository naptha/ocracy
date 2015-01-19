
function assert(x){
    if(!x) throw "Assertion failure";
}

function nanfree(m){
    for(var i = 0; i < m.length; i++){
        if(m[i].length){
            // console.log(i, m[i].length)
            for(var j = 0; j < m[i].length; j++){
                if(isNaN(m[i][j])) throw "found a NaN";
            }
        }else if(isNaN(m[i])){
            throw "found a NaN"
        }
    }
}


// NOTE: this is an ES6 function, so we should ship a polyfill
// http://jsperf.com/hyperbolic-tangent-2
// http://jsperf.com/hyperbolic-tangent
// There's actually also the chance that a pure-js solution
// ends up faster than the native one because we don't
// have to deal with the edge case of +/- Inf
function tanh(x) {
    // it's important to clip it because javascript's
    // native Math.tanh returns NaN for values of x
    // whose absolute value is greater than 400
    var y = Math.exp(2 * Math.max(-100, Math.min(100, x)));
    return (y - 1) / (y + 1);
}

// set a vector as the elementwise product of two others
function setmulvec(dest, a, b){
    for(var i = a.length; i--;) dest[i] = a[i] * b[i];
}

// add the elementwise product of two vectors to another
function addmulvec(dest, a, b){
    for(var i = a.length; i--;) dest[i] += a[i] * b[i];
}

// create a 2d matrix
function mat(r, c){
    var arr = [];
    for(var i = 0; i < r; i++) arr[i] = new Float32Array(c);
    return arr;
}

// clear a 2d matrix
function rmat(m){
    for(var i = 0; i < m.length; i++)
        for(var j = 0; j < m[i].length; j++)
            m[i][j] = NaN;
}

// returns the index of the maximal element
function max_index(n){
    var m = n[0], b = 0;
    for(var i = 1; i < n.length; i++)
        if(n[i] > m) m = n[b = i];
    return b;
}

// returns the maximal element according to a metric
function max_element(n, metric){
    var m = metric(n[0]), b = n[0];
    for(var i = 1; i < n.length; i++){
        var v = metric(n[i]);
        if(v > m){
            m = v;
            b = n[i];
        }
    }
    return b;
}


function array_split(arr, test){
    if(!test) test = function(a, b) { return a == b };
    var buf = [arr[0]],
        groups = [];
    for(var i = 1; i < arr.length; i++){
        if(!test(arr[i - 1], arr[i])){
            if(buf.length) groups.push(buf);
            buf = [];
        }
        buf.push(arr[i])
    }
    if(buf.length) groups.push(buf);
    return groups
}