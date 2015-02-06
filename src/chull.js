var canvas = document.getElementById("canvas"), 
    ctx = canvas.getContext('2d');


function rColor () {
    return '#'+Math.random().toString(16).slice(4)
}

var lobes = 5* Math.random() * Math.PI / 2
var mess = Math.random()*5
function rPoint() {
    var th = Math.random() * Math.PI * 2,
        r  = Math.random() * Math.pow(Math.sin(th * lobes)+Math.cos(mess* th * lobes), 8);
    return {
        x: 400 + Math.cos(th) * r,
        y: 400 + Math.sin(th) * r
    }
}

function drawPoints ( P ) {
    for(var i=0; i< P.length; i++){
        var p = P[i]
        ctx.beginPath()
        ctx.arc(p.x,p.y,2,0,2*Math.PI)
        ctx.fill()
    }
}

function drawPoly( P ){
    ctx.beginPath()
    ctx.moveTo(P[0].x,P[0].y)
    for(var i=1; i< P.length; i++){
        var p = P[i]
        ctx.lineTo(p.x,p.y)
    }   
    ctx.closePath()
    ctx.stroke()
}

function fillPoly( P ){
    ctx.beginPath()
    ctx.moveTo(P[0].x,P[0].y)
    for(var i=1; i< P.length; i++){
        var p = P[i]
        ctx.lineTo(p.x,p.y)
    }   
    ctx.closePath()
    ctx.fill()
}

function intersection(p1,a1,p2,a2) {
    
    var a = Math.cos(a1),
        b = - Math.sin(a1),
        c = - (a * p1.y + b * p1.x);

    var d = Math.cos(a2),
        e = - Math.sin(a2),
        f = - (d * p2.y + e * p2.x);

    var denom = -b*d+a*e
    return { x:(c*d-a*f)/denom, y:(-c*e+b*f)/denom}
}

function dist_line_point(origin,angle,p) {
    var b = -1,
        a = Math.tan(angle),
        c = origin.y - a*origin.x;
    return Math.abs(a*p.x + b*p.y + c)/Math.sqrt(a*a + b*b)
}

function cross(o, a, b) {
   return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}
 

function convexHull(points) {
   points.sort(function(a, b) {
      return a.x == b.x ? a.y - b.y : a.x - b.x;
   })
 
   var lower = [];
   for (var i = 0; i < points.length; i++) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], points[i]) <= 0)
         lower.pop();
      lower.push(points[i]);
   }
 
   var upper = [];
   for (var i = points.length - 1; i >= 0; i--) {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], points[i]) <= 0)
         upper.pop();
      upper.push(points[i]);
   }
 
   upper.pop();
   lower.pop();
   return lower.concat(upper);
}

function min_enclosing_rect( poly ){

    var minx = 0,
        miny = 0,
        maxx = 0,
        maxy = 0, 
        numcals = 4;

    for (var i = 1; i < poly.length; i++) {
        var p = poly[i]
        if (p.x < poly[minx].x) minx = i;
        if (p.y < poly[miny].y) miny = i;
        if (p.x > poly[maxx].x) maxx = i;
        if (p.y > poly[maxy].y) maxy = i;
    }

    var cal    = [], 
        points = [miny, maxx, maxy, minx];

    var dimension = function(n){ return dist_line_point(poly[cal[n].point], cal[n].angle, poly[cal[n+2].point]) }
    var corner = function(n){ return intersection(poly[cal[n].point], cal[n].angle, poly[cal[(n+1) % numcals].point], cal[(n+1) % numcals].angle) }

    for (var i = 0; i < numcals; i++) cal.push({ point: points[i], angle: i * Math.PI/2 })

    var pminx = poly[minx], 
        pminy = poly[miny], 
        pmaxx = poly[maxx], 
        pmaxy = poly[maxy],
        totalAngle  = 0,
        minArea = (pmaxx.x - pminx.x) * (pmaxy.y - pminy.y),
        minRect = [
            {x: pmaxx.x, y: pmaxy.y},
            {x: pminx.x, y: pmaxy.y},
            {x: pminx.x, y: pminy.y},
            {x: pmaxx.x, y: pminy.y},
        ];

    while (totalAngle <= Math.PI/2) {
        var minAngle = Infinity,
            minCal = 0;

        for (var i = 0; i < numcals; i++) {
            var c = cal[i],
                p = poly[(c.point + 1) % poly.length],
                cp = poly[c.point],
                angle = Math.atan2(p.y - cp.y, p.x - cp.x) - c.angle;
            
            while (angle < 0) angle += Math.PI * 2;



            if (angle < minAngle) {
                minAngle = angle
                minCal = i
            }
        }

        cal[minCal].point = (cal[minCal].point + 1) % poly.length;

        for (var i = 0; i < numcals; i++) cal[i].angle += minAngle;

        var area = dimension(0) * dimension(1)
        if(area < minArea){
            
            ctx.strokeStyle = rColor()
  
            for(var i = 0; i<numcals; i++){
                var cp = poly[cal[i].point]
                var angle = cal[i].angle
            }

            minArea = area
            minRect = [0, 1, 2, 3].map(corner)
        }
        totalAngle += minAngle
    }
    return minRect
}