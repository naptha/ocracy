// var lineheight = 28,
//     hiddensize = 50;

var hiddensize = fwdWGI.length,
    lineheight = fwdWGI[0].length - hiddensize - 1;

codec = [""," ","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","{","|","}","~"]

net = new SequenceRecognizer(lineheight, hiddensize, codec);

var parallel = net.lstm.nets[0],
    softmax  = net.lstm.nets[1],
    fwd      = parallel.nets[0],
    rev      = parallel.nets[1].net;


var dict = '';
for(var i = 32; i < 126; i++){
    if(i == 92 || i == 34 || i == 39 || i == 60) continue;
    dict += String.fromCharCode(i)
}

function vecparse(str){
    if(typeof str != 'string') return str;
  var arr = new Float32Array(str.length);
    for(var i = 0; i < str.length; i++)
        arr[i] = 2 * encGamut * dict.indexOf(str[i]) / (dict.length - 1) - encGamut;
    return arr;
}

fwd.WGI = fwdWGI.map(vecparse)
fwd.WGF = fwdWGF.map(vecparse)
fwd.WGO = fwdWGO.map(vecparse)
fwd.WCI = fwdWCI.map(vecparse)
fwd.WIP = vecparse(fwdWIP);
fwd.WFP = vecparse(fwdWFP);
fwd.WOP = vecparse(fwdWOP);

rev.WGI = revWGI.map(vecparse)
rev.WGF = revWGF.map(vecparse)
rev.WGO = revWGO.map(vecparse)
rev.WCI = revWCI.map(vecparse)
rev.WIP = vecparse(revWIP);
rev.WFP = vecparse(revWFP);
rev.WOP = vecparse(revWOP);

softmax.W = softW.map(vecparse)
softmax.B = vecparse(softB);


function displayImage(image){
    var canvas = document.createElement('canvas')
    var imgwidth = image.length,
        imgheight = image[0].length;
    canvas.width = imgwidth;
    canvas.height = imgheight;
    var ctx = canvas.getContext('2d');
    var img = ctx.createImageData(imgwidth, imgheight);
    for(var i = 0; i < imgwidth; i++){
        for(var j = 0; j < imgheight; j++){
            var o = j * imgwidth + i;
            img.data[4 * o + 0] = img.data[4 * o + 1] = img.data[4 * o + 2] = Math.floor(image[i][j] * 255);
            img.data[4 * o + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0)
    document.body.appendChild(canvas)
}


function visualizeOutput(output){
    var canvas = document.createElement('canvas')
    var ctx = canvas.getContext('2d');
    var imgwidth = output.length,
        imgheight = output[0].length;
    canvas.width = imgwidth;
    canvas.height = imgheight;
    // console.log(imgwidth, imgheight)
    var img = ctx.createImageData(imgwidth, imgheight);
    for(var i = 0; i < imgwidth; i++){
        for(var j = 0; j < imgheight; j++){
            var o = j * imgwidth + i;
            img.data[4 * o + 0] = img.data[4 * o + 1] = img.data[4 * o + 2] = Math.floor(output[i][j] * 255);
            img.data[4 * o + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0)
    for(var j = 0; j < imgheight; j++){
        var run_start = -1;
        for(var i = 0; i < imgwidth; i++){
            if(output[i][j] > 0.1){
                if(run_start < 0){
                    run_start = i
                }
            }else{
                if(run_start >= 0){
                    ctx.fillStyle = 'green'
                    ctx.font = '7px sans-serif'
                    ctx.textAlign = 'center'
                    ctx.fillText(codec[j], i / 2 + run_start / 2, j - 2)
                    run_start = -1
                }
            }
        }
    }
    document.body.appendChild(canvas)

}

function sparkline(net){
    var canvas = document.createElement('canvas')
    var ctx = canvas.getContext('2d');
    var imgwidth = net.output.length,
        imgheight = net.output[0].length;
    canvas.width = imgwidth;
    canvas.height = 100;
    // console.log(imgwidth, imgheight)
    // var img = ctx.createImageData(imgwidth, imgheight);
    ctx.beginPath()
    // var merp = array_conv3(net.output.map(function(e){ return 1 - e[0] }), function(a, b, c){
    //     // return Math.max(a, b, c) * 0.5 + b * 0.5
    //     // return Math.exp(b) / (Math.exp(a) + Math.exp(b) + Math.exp(c))
    //     // return Math.max(a, b, c)
    //     // return 0.1 * a + 0.8 * b + 0.1 * c;
    //     return Math.max(a, b) / 2 + Math.max(b, c) / 2
    //     // return Math.max(b, c)
    //     // return Math.max(a, b, c) / 2 + Math.min(a, b, c) / 2
    //     // return b
    // })



    // for(var i = 1; i < net.output.length; i++){
    //     var sum = 0;
    //     for(var j = 1; j < net.output[0].length; j++){
    //         sum += Math.abs(net.output[i - 1][j] - net.output[i][j])
    //     }
    //     // console.log(sum)
    //     merp[i] *= 1-sum;
    //     // if(sum > 0.5){
    //         // merp[i] = 0
    //         // merp[i] *= (1-sum)
    //     // }
    // }

    // var derp = array_conv3(merp, function(a, b, c){
    //     // return Math.max(a, b, c) * 0.5 + b * 0.5
    //     // return Math.exp(b) / (Math.exp(a) + Math.exp(b) + Math.exp(c))
    //     // return Math.max(a, b, c)
    //     return 0.1 * a + 0.8 * b + 0.1 * c;
    //     // return Math.max(a, b) / 2 + Math.max(b, c) / 2
    //     // return Math.max(b, c)
    //     // return Math.max(a, b, c) / 2 + Math.min(a, b, c) / 2
    //     // return b
    // })



    for(var i = 0; i < imgwidth; i++) ctx.lineTo(i, 100 - 100 * net.output[i][0]);
    ctx.strokeStyle = 'orange'
    ctx.stroke()

    ctx.beginPath()
    for(var i = 0; i < imgwidth; i++) ctx.lineTo(i, 100 - 100 * net.output[i][1]);
    ctx.strokeStyle = 'green'
    ctx.stroke()

    // ctx.beginPath()
    // for(var i = 0; i < imgwidth; i++) ctx.lineTo(i, 100 - 100 * Array.prototype.slice.call(net.output[i], 2).reduce(function(a, b){ return a + b }));
    // ctx.strokeStyle = 'rgba(0,0,255,0.5)'
    // ctx.stroke()

    document.body.appendChild(canvas)

}

var outputs = []

function recognize(){
    console.time('recognition start');
    [img010001, img010002, img010003, img010004, img010005, img010006, img010007, img010008, img010009, img01000a, img01000b, img01000c, img01000d, img01000e, img01000f, img010010, img010011, img010012, img010013, img010014, img010015, img010016, img010017, img010018, img010019, img01001a].forEach(function(image, i){

        document.body.appendChild(document.createTextNode(i))
        document.body.appendChild(document.createElement('br'))
        displayImage(image);
        console.log(net.predictString(image))
        document.body.appendChild(document.createElement('br'))
        // displayImage(net.output)
        visualizeOutput(net.output);
        document.body.appendChild(document.createElement('br'))
        sparkline(net)
        document.body.appendChild(document.createElement('br'))

        outputs[i] = net.output
    })
    console.timeEnd('recognition start')

}

// function f(e){return 1 - e[0] > 0.4}; array_split(outputs[9], function(a, b){return f(a) == f(b)}).filter(function(e){return f(e[0])}).map(function(e){return e.map(function(k){return codec[max_index(k)]}).join('')})


// function f(e){return 1 - e[0] > 0.7}; array_split(outputs[9], function(a, b){return f(a) == f(b)}).filter(function(e){return f(e[0])}).map(function(e){return codec[max_index(array_zip(e).map(array_mean))]}).join('')

// array_split(outputs[18], function(a, b){
//         return f(a) == f(b)
//     }).filter(function(e){
// return e.map(function(k){
// return 1 -k[0]
// }).reduce(function(a,b){return a + b}) > 0.8
// }).map(function(e){
//         return max_index(array_zip(e).map(array_mean))
//     }).map(function(e){return codec[e]}).join('')

onload = function(){
    recognize()
}