
// canny & sobel dx/dy
function raw_swt(img_canny, img_dxdy, params){
	var max_stroke = params.max_stroke, // maximum stroke width
		direction = params.direction,
		width = img_canny.cols,
		height = img_canny.rows;

	// nonzero Math.min function, if a is zero, returns b, otherwise minimizes
	function nzmin(a, b){
		if(a === 0) return b;
		if(a < b) return a;
		return b;
	}
	
	var strokes = [];
	var swt = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	
	console.time("first pass")
	// first pass of stroke width transform 
	for(var i = 0; i < width * height; i++){
		if(img_canny.data[i] != 0xff) continue; // only apply on edge pixels

		var itheta = Math.atan2(img_dxdy.data[(i<<1) + 1], img_dxdy.data[i<<1]); // calculate the image gradient at this point by sobel
		var ray = [i];
		var step = 1;
		
		var ix = i % width, iy = Math.floor(i / width);
		while(step < max_stroke){
			// extrapolate the ray outwards depending on search direction
			// libccv is particularly clever in that it uses 
			// bresenham's line drawing algorithm to pick out
			// the points along the line and also checks 
			// neighboring pixels for corners

			var jx = Math.round(ix + Math.cos(itheta) * direction * step);
			var jy = Math.round(iy + Math.sin(itheta) * direction * step);
			step++;
			if(jx < 0 || jy < 0 || jx > width || jy > height) break;
			var j = jy * width + jx;
			ray.push(j)
			if(img_canny.data[j] != 0xff) continue;
			// calculate theta for this ray since we've reached the other side
			var jtheta = Math.atan2(img_dxdy.data[(j<<1) + 1], img_dxdy.data[j<<1]); 
			
			if(Math.abs(Math.abs(itheta - jtheta) - Math.PI) < Math.PI / 2){ // check if theta reflects the starting angle approximately
				strokes.push(i)
				var sw = Math.sqrt((jx - ix) * (jx - ix) + (jy - iy) * (jy - iy)) // derive the stroke width
				for(var k = 0; k < ray.length; k++){ // iterate rays and set points along ray to minimum stroke width
					swt.data[ray[k]] = nzmin(swt.data[ray[k]], sw) // use nzmin because it's initially all 0's
				}
			}
			break;
		}
	}
	console.timeEnd("first pass")
	console.time("refinement pass")

	// second pass, refines swt values as median
	for(var k = 0; k < strokes.length; k++){
		var i = strokes[k];
		var itheta = Math.atan2(img_dxdy.data[(i<<1) + 1], img_dxdy.data[i<<1]);
		var ray = [];
		var widths = []
		var step = 1;

		var ix = i % width, iy = Math.floor(i / width);
		while(step < max_stroke){
			var jx = Math.round(ix + Math.cos(itheta) * direction * step);
			var jy = Math.round(iy + Math.sin(itheta) * direction * step);
			step++;
			var j = jy * width + jx;
			// record position of the ray and the stroke width there
			widths.push(swt.data[j])
			ray.push(j)			
			// stop when the ray is terminated
			if(img_canny.data[j] == 0xff) break;
		}
		var median = widths.sort(function(a, b){return a - b})[Math.floor(widths.length / 2)];
		// set the high values to the median so that corners are nice
		for(var j = 0; j < ray.length; j++){
			swt.data[ray[j]] = nzmin(swt.data[ray[j]], median)
		}
		// swt.data[ray[0]] = 0
		// swt.data[ray[ray.length - 1]] = 0
	}

	console.timeEnd("refinement pass")
	
	// TODO: get rid of strokes
	return {
		swt: swt,
		strokes: strokes
	}
}



// maybe in the future we should replace this with a strongly
// connected components algorithm (or have some spatial heuristic to
// determine how wise it would be to consider the connection valid)
function connected_swt(swt, params){
	var dx8 = [-1, 1, -1, 0, 1, -1, 0, 1];
	var dy8 = [0, 0, -1, -1, -1, 1, 1, 1];
	var width = swt.cols, 
		height = swt.rows;

	var marker = new jsfeat.matrix_t(width, height, jsfeat.U8C1_t)
	var contours = []
	
	for(var i = 0; i < width * height; i++){
		if(marker.data[i] || !swt.data[i]) continue;

		var ix = i % width, iy = Math.floor(i / width)
		
		marker.data[i] = 1
		var contour = []
		var stack = [i]
		var closed;
		
		while(closed = stack.shift()){
			contour.push(closed)
			var cx = closed % width, cy = Math.floor(closed / width);
			var w = swt.data[closed];
			for(var k = 0; k < 8; k++){
				var nx = cx + dx8[k]
				var ny = cy + dy8[k]
				var n = ny * width + nx;

				if(nx >= 0 && nx < width &&
				   ny >= 0 && ny < height &&
				   swt.data[n] &&
				   !marker.data[n] &&
				   swt.data[n] <= params.stroke_ratio * w &&
				   swt.data[n] * params.stroke_ratio >= w){
					marker.data[n] = 1
					// update the average stroke width
					w = (w * stack.length + swt.data[n]) / (stack.length + 1)
					stack.push(n)
				}
			}
		}
		// contours.push(contour)
		if(contour.length >= params.min_area){
			contours.push(contour)	
		}
	}
	return contours
}



function visualize_matrix(mat, letters){
	var c = document.createElement('canvas')
	c.width = mat.cols;
	c.height = mat.rows;
	var cx = c.getContext('2d')
	var out = cx.createImageData(mat.cols, mat.rows);
	for(var i = 0; i < mat.rows * mat.cols; i++){

		out.data[i * 4 + 3] = 255
		if(mat.data[i] == 1){
			out.data[i * 4] = 255
			// out.data[i * 4 + 1] = out.data[i * 4 + 2] = 30 * mat.data[i]
		}else{
			out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = 30 * mat.data[i]	
		}
		
	}
	cx.putImageData(out, 0, 0)
	
	if(letters){
		cx.strokeStyle = 'red'
		for(var i = 0; i < letters.length; i++){
			var letter = letters[i];
			cx.strokeRect(letter.x0 + .5, letter.y0 + .5, letter.width, letter.height)
		}

		if(letters[0] && letters[0].letters){
			// hey look not actually letters
			letters.forEach(function(line){
				cx.beginPath()
				var colors = ['green', 'blue', 'red', 'purple', 'orange', 'yellow']
				cx.strokeStyle = colors[Math.floor(colors.length * Math.random())]
				cx.lineWidth = 3

				line.letters
					.sort(function(a, b){ return a.cx - b.cx })
					.forEach(function(letter){
						cx.lineTo(letter.cx, letter.cy)
					})

				cx.stroke()
			})
		}
	}
	document.body.appendChild(c)

	// console.image(c.toDataURL('image/png'))

	return cx
}


function equivalence_classes(elements, is_equal){
	var node = []
	for(var i = 0; i < elements.length; i++){
		node.push({
			parent: 0,
			element: elements[i],
			rank: 0
		})
	}
	for(var i = 0; i < node.length; i++){
		var root = node[i]
		while(root.parent){
			root = root.parent;
		}
		for(var j = 0; j < node.length; j++){
			if(i == j) continue;
			if(!is_equal(node[i].element, node[j].element)) continue;
			var root2 = node[j];
			while(root2.parent){
				root2 = root2.parent;
			}
			if(root2 != root){
				if(root.rank > root2.rank){
					root2.parent = root;
				}else{
					root.parent = root2;
					if(root.rank == root2.rank){
						root2.rank++  
					}
					root = root2;
				}
				var node2 = node[j];
				while(node2.parent){
					var temp = node2;
					node2 = node2.parent;
					temp.parent = root;
				}
				var node2 = node[i];
				while(node2.parent){
					var temp = node2;
					node2 = node2.parent;
					temp.parent = root;
				}
			}
		}
	}
	var index = 0;
	var clusters = [];
	for(var i = 0; i < node.length; i++){
		var j = -1;
		var node1 = node[i]
		while(node1.parent){
			node1 = node1.parent
		}
		if(node1.rank >= 0){
			node1.rank = ~index++;
		}
		j = ~node1.rank;

		if(clusters[j]){
			clusters[j].push(elements[i])
		}else{
			clusters[j] = [elements[i]]
		}
	}
	return clusters;
}