// https://github.com/naptha/jsfeat/blob/master/build/jsfeat-custom.js


/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

// namespace ?
var jsfeat = jsfeat || { REVISION: 'ALPHA' };



/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

(function(global) {
    "use strict";
    //

    // CONSTANTS
    var EPSILON = 0.0000001192092896;
    var FLT_MIN = 1E-37;

    // implementation from CCV project
    // currently working only with u8,s32,f32
    var U8_t = 0x0100,
        S32_t = 0x0200,
        F32_t = 0x0400,
        S64_t = 0x0800,
        F64_t = 0x1000;

    var C1_t = 0x01,
        C2_t = 0x02,
        C3_t = 0x03,
        C4_t = 0x04;

    var _data_type_size = new Int32Array([ -1, 1, 4, -1, 4, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, 8 ]);

    var get_data_type = (function () {
        return function(type) {
            return (type & 0xFF00);
        }
    })();

    var get_channel = (function () {
        return function(type) {
            return (type & 0xFF);
        }
    })();

    var get_data_type_size = (function () {
        return function(type) {
            return _data_type_size[(type & 0xFF00) >> 8];
        }
    })();

    // box blur option
    var BOX_BLUR_NOSCALE = 0x01;
    // svd options
    var SVD_U_T = 0x01;
    var SVD_V_T = 0x02;

    var data_t = (function () {
        function data_t(size_in_bytes, buffer) {
            // we need align size to multiple of 8
            this.size = ((size_in_bytes + 7) | 0) & -8;
            if (typeof buffer === "undefined") { 
                this.buffer = new ArrayBuffer(this.size);
            } else {
                this.buffer = buffer;
                this.size = buffer.length;
            }
            this.u8 = new Uint8Array(this.buffer);
            this.i32 = new Int32Array(this.buffer);
            this.f32 = new Float32Array(this.buffer);
            this.f64 = new Float64Array(this.buffer);
        }
        return data_t;
    })();

    var matrix_t = (function () {
        // columns, rows, data_type
        function matrix_t(c, r, data_type, data_buffer) {
            this.type = get_data_type(data_type)|0;
            this.channel = get_channel(data_type)|0;
            this.cols = c|0;
            this.rows = r|0;
            if (typeof data_buffer === "undefined") { 
                this.allocate();
            } else {
                this.buffer = data_buffer;
                // data user asked for
                this.data = this.type&U8_t ? this.buffer.u8 : (this.type&S32_t ? this.buffer.i32 : (this.type&F32_t ? this.buffer.f32 : this.buffer.f64));
            }
        }
        matrix_t.prototype.allocate = function() {
            // clear references
            delete this.data;
            delete this.buffer;
            //
            this.buffer = new data_t((this.cols * get_data_type_size(this.type) * this.channel) * this.rows);
            this.data = this.type&U8_t ? this.buffer.u8 : (this.type&S32_t ? this.buffer.i32 : (this.type&F32_t ? this.buffer.f32 : this.buffer.f64));
        }
        matrix_t.prototype.copy_to = function(other) {
            var od = other.data, td = this.data;
            var i = 0, n = (this.cols*this.rows*this.channel)|0;
            for(; i < n-4; i+=4) {
                od[i] = td[i];
                od[i+1] = td[i+1];
                od[i+2] = td[i+2];
                od[i+3] = td[i+3];
            }
            for(; i < n; ++i) {
                od[i] = td[i];
            }
        }
        matrix_t.prototype.resize = function(c, r, ch) {
            if (typeof ch === "undefined") { ch = this.channel; }
            // change buffer only if new size doesnt fit
            var new_size = (c * ch) * r;
            if(new_size > this.rows*this.cols*this.channel) {
                this.cols = c;
                this.rows = r;
                this.channel = ch;
                this.allocate();
            } else {
                this.cols = c;
                this.rows = r;
                this.channel = ch;
            }
        }

        return matrix_t;
    })();

    var pyramid_t = (function () {

        function pyramid_t(levels) {
            this.levels = levels|0;
            this.data = new Array(levels);
            this.pyrdown = jsfeat.imgproc.pyrdown;
        }

        pyramid_t.prototype.allocate = function(start_w, start_h, data_type) {
            var i = this.levels;
            while(--i >= 0) {
                this.data[i] = new matrix_t(start_w >> i, start_h >> i, data_type);
            }
        }

        pyramid_t.prototype.build = function(input, skip_first_level) {
            if (typeof skip_first_level === "undefined") { skip_first_level = true; }
            // just copy data to first level
            var i = 2, a = input, b = this.data[0];
            if(!skip_first_level) {
                var j=input.cols*input.rows;
                while(--j >= 0) {
                    b.data[j] = input.data[j];
                }
            }
            b = this.data[1];
            this.pyrdown(a, b);
            for(; i < this.levels; ++i) {
                a = b;
                b = this.data[i];
                this.pyrdown(a, b);
            }
        }

        return pyramid_t;
    })();

    var point2d_t = (function () {
        function point2d_t(x,y,score,level) {
            if (typeof x === "undefined") { x=0; }
            if (typeof y === "undefined") { y=0; }
            if (typeof score === "undefined") { score=0; }
            if (typeof level === "undefined") { level=0; }

            this.x = x;
            this.y = y;
            this.score = score;
            this.level = level;
        }
        return point2d_t;
    })();


    // data types
    global.U8_t = U8_t;
    global.S32_t = S32_t;
    global.F32_t = F32_t;
    global.S64_t = S64_t;
    global.F64_t = F64_t;
    // data channels
    global.C1_t = C1_t;
    global.C2_t = C2_t;
    global.C3_t = C3_t;
    global.C4_t = C4_t;

    // popular formats
    global.U8C1_t = U8_t | C1_t;
    global.U8C3_t = U8_t | C3_t;
    global.U8C4_t = U8_t | C4_t;

    global.F32C1_t = F32_t | C1_t;
    global.F32C2_t = F32_t | C2_t;
    global.S32C1_t = S32_t | C1_t;
    global.S32C2_t = S32_t | C2_t;

    // constants
    global.EPSILON = EPSILON;
    global.FLT_MIN = FLT_MIN;

    // options
    global.BOX_BLUR_NOSCALE = BOX_BLUR_NOSCALE;
    global.SVD_U_T = SVD_U_T;
    global.SVD_V_T = SVD_V_T;

    global.get_data_type = get_data_type;
    global.get_channel = get_channel;
    global.get_data_type_size = get_data_type_size;

    global.data_t = data_t;
    global.matrix_t = matrix_t;
    global.pyramid_t = pyramid_t;
    global.point2d_t = point2d_t;

})(jsfeat);



/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

(function(global) {
    "use strict";
    //

    var cache = (function() {

        // very primitive array cache, still need testing if it helps
        // of course V8 has its own powerful cache sys but i'm not sure
        // it caches several multichannel 640x480 buffer creations each frame

        var _pool_node_t = (function () {
            function _pool_node_t(size_in_bytes) {
                this.next = null;
                this.data = new jsfeat.data_t(size_in_bytes);
                this.size = this.data.size;
                this.buffer = this.data.buffer;
                this.u8 = this.data.u8;
                this.i32 = this.data.i32;
                this.f32 = this.data.f32;
                this.f64 = this.data.f64;
            }
            _pool_node_t.prototype.resize = function(size_in_bytes) {
                delete this.data;
                this.data = new jsfeat.data_t(size_in_bytes);
                this.size = this.data.size;
                this.buffer = this.data.buffer;
                this.u8 = this.data.u8;
                this.i32 = this.data.i32;
                this.f32 = this.data.f32;
                this.f64 = this.data.f64;
            }
            return _pool_node_t;
        })();

        var _pool_head, _pool_tail;
        var _pool_size = 0;

        return {

            allocate: function(capacity, data_size) {
                _pool_head = _pool_tail = new _pool_node_t(data_size);
                for (var i = 0; i < capacity; ++i) {
                    var node = new _pool_node_t(data_size);
                    _pool_tail = _pool_tail.next = node;

                    _pool_size++;
                }
            },

            get_buffer: function(size_in_bytes) {
                // assume we have enough free nodes
                var node = _pool_head;
                _pool_head = _pool_head.next;
                _pool_size--;

                if(size_in_bytes > node.size) {
                    node.resize(size_in_bytes);
                }

                return node;
            },

            put_buffer: function(node) {
                _pool_tail = _pool_tail.next = node;
                _pool_size++;
            }
        };
    })();

    global.cache = cache;
    // for now we dont need more than 30 buffers
    // if having cache sys really helps we can add auto extending sys
    cache.allocate(30, 640*4);

})(jsfeat);


/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

(function(global) {
    "use strict";
    //

    var math = (function() {

        var qsort_stack = new Int32Array(48*2);

        return {
            get_gaussian_kernel: function(size, sigma, kernel, data_type) {
                var i=0,x=0.0,t=0.0,sigma_x=0.0,scale_2x=0.0;
                var sum = 0.0;
                var kern_node = jsfeat.cache.get_buffer(size<<2);
                var _kernel = kern_node.f32;//new Float32Array(size);

                if((size&1) == 1 && size <= 7 && sigma <= 0) {
                    switch(size>>1) {
                        case 0:
                        _kernel[0] = 1.0;
                        sum = 1.0;
                        break;
                        case 1:
                        _kernel[0] = 0.25, _kernel[1] = 0.5, _kernel[2] = 0.25;
                        sum = 0.25+0.5+0.25;
                        break;
                        case 2:
                        _kernel[0] = 0.0625, _kernel[1] = 0.25, _kernel[2] = 0.375, 
                        _kernel[3] = 0.25, _kernel[4] = 0.0625;
                        sum = 0.0625+0.25+0.375+0.25+0.0625;
                        break;
                        case 3:
                        _kernel[0] = 0.03125, _kernel[1] = 0.109375, _kernel[2] = 0.21875, 
                        _kernel[3] = 0.28125, _kernel[4] = 0.21875, _kernel[5] = 0.109375, _kernel[6] = 0.03125;
                        sum = 0.03125+0.109375+0.21875+0.28125+0.21875+0.109375+0.03125;
                        break;
                    }
                } else {
                    sigma_x = sigma > 0 ? sigma : ((size-1)*0.5 - 1.0)*0.3 + 0.8;
                    scale_2x = -0.5/(sigma_x*sigma_x);

                    for( ; i < size; ++i )
                    {
                        x = i - (size-1)*0.5;
                        t = Math.exp(scale_2x*x*x);

                        _kernel[i] = t;
                        sum += t;
                    }
                }

                if(data_type & jsfeat.U8_t) {
                    // int based kernel
                    sum = 256.0/sum;
                    for (i = 0; i < size; ++i) {
                        kernel[i] = (_kernel[i] * sum + 0.5)|0;
                    }
                } else {
                    // classic kernel
                    sum = 1.0/sum;
                    for (i = 0; i < size; ++i) {
                        kernel[i] = _kernel[i] * sum;
                    }
                }

                jsfeat.cache.put_buffer(kern_node);
            },

            // model is 3x3 matrix_t
            perspective_4point_transform: function(model, src_x0, src_y0, dst_x0, dst_y0,
                                                        src_x1, src_y1, dst_x1, dst_y1,
                                                        src_x2, src_y2, dst_x2, dst_y2,
                                                        src_x3, src_y3, dst_x3, dst_y3) {
                var t1 = src_x0;
                var t2 = src_x2;
                var t4 = src_y1;
                var t5 = t1 * t2 * t4;
                var t6 = src_y3;
                var t7 = t1 * t6;
                var t8 = t2 * t7;
                var t9 = src_y2;
                var t10 = t1 * t9;
                var t11 = src_x1;
                var t14 = src_y0;
                var t15 = src_x3;
                var t16 = t14 * t15;
                var t18 = t16 * t11;
                var t20 = t15 * t11 * t9;
                var t21 = t15 * t4;
                var t24 = t15 * t9;
                var t25 = t2 * t4;
                var t26 = t6 * t2;
                var t27 = t6 * t11;
                var t28 = t9 * t11;
                var t30 = 1.0 / (t21-t24 - t25 + t26 - t27 + t28);
                var t32 = t1 * t15;
                var t35 = t14 * t11;
                var t41 = t4 * t1;
                var t42 = t6 * t41;
                var t43 = t14 * t2;
                var t46 = t16 * t9;
                var t48 = t14 * t9 * t11;
                var t51 = t4 * t6 * t2;
                var t55 = t6 * t14;
                var Hr0 = -(t8-t5 + t10 * t11 - t11 * t7 - t16 * t2 + t18 - t20 + t21 * t2) * t30;
                var Hr1 = (t5 - t8 - t32 * t4 + t32 * t9 + t18 - t2 * t35 + t27 * t2 - t20) * t30;
                var Hr2 = t1;
                var Hr3 = (-t9 * t7 + t42 + t43 * t4 - t16 * t4 + t46 - t48 + t27 * t9 - t51) * t30;
                var Hr4 = (-t42 + t41 * t9 - t55 * t2 + t46 - t48 + t55 * t11 + t51 - t21 * t9) * t30;
                var Hr5 = t14;
                var Hr6 = (-t10 + t41 + t43 - t35 + t24 - t21 - t26 + t27) * t30;
                var Hr7 = (-t7 + t10 + t16 - t43 + t27 - t28 - t21 + t25) * t30;
                
                t1 = dst_x0;
                t2 = dst_x2;
                t4 = dst_y1;
                t5 = t1 * t2 * t4;
                t6 = dst_y3;
                t7 = t1 * t6;
                t8 = t2 * t7;
                t9 = dst_y2;
                t10 = t1 * t9;
                t11 = dst_x1;
                t14 = dst_y0;
                t15 = dst_x3;
                t16 = t14 * t15;
                t18 = t16 * t11;
                t20 = t15 * t11 * t9;
                t21 = t15 * t4;
                t24 = t15 * t9;
                t25 = t2 * t4;
                t26 = t6 * t2;
                t27 = t6 * t11;
                t28 = t9 * t11;
                t30 = 1.0 / (t21-t24 - t25 + t26 - t27 + t28);
                t32 = t1 * t15;
                t35 = t14 * t11;
                t41 = t4 * t1;
                t42 = t6 * t41;
                t43 = t14 * t2;
                t46 = t16 * t9;
                t48 = t14 * t9 * t11;
                t51 = t4 * t6 * t2;
                t55 = t6 * t14;
                var Hl0 = -(t8-t5 + t10 * t11 - t11 * t7 - t16 * t2 + t18 - t20 + t21 * t2) * t30;
                var Hl1 = (t5 - t8 - t32 * t4 + t32 * t9 + t18 - t2 * t35 + t27 * t2 - t20) * t30;
                var Hl2 = t1;
                var Hl3 = (-t9 * t7 + t42 + t43 * t4 - t16 * t4 + t46 - t48 + t27 * t9 - t51) * t30;
                var Hl4 = (-t42 + t41 * t9 - t55 * t2 + t46 - t48 + t55 * t11 + t51 - t21 * t9) * t30;
                var Hl5 = t14;
                var Hl6 = (-t10 + t41 + t43 - t35 + t24 - t21 - t26 + t27) * t30;
                var Hl7 = (-t7 + t10 + t16 - t43 + t27 - t28 - t21 + t25) * t30;

                // the following code computes R = Hl * inverse Hr
                t2 = Hr4-Hr7*Hr5;
                t4 = Hr0*Hr4;
                t5 = Hr0*Hr5;
                t7 = Hr3*Hr1;
                t8 = Hr2*Hr3;
                t10 = Hr1*Hr6;
                var t12 = Hr2*Hr6;
                t15 = 1.0 / (t4-t5*Hr7-t7+t8*Hr7+t10*Hr5-t12*Hr4);
                t18 = -Hr3+Hr5*Hr6;
                var t23 = -Hr3*Hr7+Hr4*Hr6;
                t28 = -Hr1+Hr2*Hr7;
                var t31 = Hr0-t12;
                t35 = Hr0*Hr7-t10;
                t41 = -Hr1*Hr5+Hr2*Hr4;
                var t44 = t5-t8;
                var t47 = t4-t7;
                t48 = t2*t15;
                var t49 = t28*t15;
                var t50 = t41*t15;
                var mat = model.data;
                mat[0] = Hl0*t48+Hl1*(t18*t15)-Hl2*(t23*t15);
                mat[1] = Hl0*t49+Hl1*(t31*t15)-Hl2*(t35*t15);
                mat[2] = -Hl0*t50-Hl1*(t44*t15)+Hl2*(t47*t15);
                mat[3] = Hl3*t48+Hl4*(t18*t15)-Hl5*(t23*t15);
                mat[4] = Hl3*t49+Hl4*(t31*t15)-Hl5*(t35*t15);
                mat[5] = -Hl3*t50-Hl4*(t44*t15)+Hl5*(t47*t15);
                mat[6] = Hl6*t48+Hl7*(t18*t15)-t23*t15;
                mat[7] = Hl6*t49+Hl7*(t31*t15)-t35*t15;
                mat[8] = -Hl6*t50-Hl7*(t44*t15)+t47*t15;
            },

            // The current implementation was derived from *BSD system qsort():
            // Copyright (c) 1992, 1993
            // The Regents of the University of California.  All rights reserved.
            qsort: function(array, low, high, cmp) {
                var isort_thresh = 7;
                var t,ta,tb,tc;
                var sp = 0,left=0,right=0,i=0,n=0,m=0,ptr=0,ptr2=0,d=0;
                var left0=0,left1=0,right0=0,right1=0,pivot=0,a=0,b=0,c=0,swap_cnt=0;

                var stack = qsort_stack;

                if( (high-low+1) <= 1 ) return;

                stack[0] = low;
                stack[1] = high;

                while( sp >= 0 ) {
                
                    left = stack[sp<<1];
                    right = stack[(sp<<1)+1];
                    sp--;

                    for(;;) {
                        n = (right - left) + 1;

                        if( n <= isort_thresh ) {
                        //insert_sort:
                            for( ptr = left + 1; ptr <= right; ptr++ ) {
                                for( ptr2 = ptr; ptr2 > left && cmp(array[ptr2],array[ptr2-1]); ptr2--) {
                                    t = array[ptr2];
                                    array[ptr2] = array[ptr2-1];
                                    array[ptr2-1] = t;
                                }
                            }
                            break;
                        } else {
                            swap_cnt = 0;

                            left0 = left;
                            right0 = right;
                            pivot = left + (n>>1);

                            if( n > 40 ) {
                                d = n >> 3;
                                a = left, b = left + d, c = left + (d<<1);
                                ta = array[a],tb = array[b],tc = array[c];
                                left = cmp(ta, tb) ? (cmp(tb, tc) ? b : (cmp(ta, tc) ? c : a))
                                                  : (cmp(tc, tb) ? b : (cmp(ta, tc) ? a : c));

                                a = pivot - d, b = pivot, c = pivot + d;
                                ta = array[a],tb = array[b],tc = array[c];
                                pivot = cmp(ta, tb) ? (cmp(tb, tc) ? b : (cmp(ta, tc) ? c : a))
                                                  : (cmp(tc, tb) ? b : (cmp(ta, tc) ? a : c));

                                a = right - (d<<1), b = right - d, c = right;
                                ta = array[a],tb = array[b],tc = array[c];
                                right = cmp(ta, tb) ? (cmp(tb, tc) ? b : (cmp(ta, tc) ? c : a))
                                                  : (cmp(tc, tb) ? b : (cmp(ta, tc) ? a : c));
                            }

                            a = left, b = pivot, c = right;
                            ta = array[a],tb = array[b],tc = array[c];
                            pivot = cmp(ta, tb) ? (cmp(tb, tc) ? b : (cmp(ta, tc) ? c : a))   
                                               : (cmp(tc, tb) ? b : (cmp(ta, tc) ? a : c));
                            if( pivot != left0 ) {
                                t = array[pivot];
                                array[pivot] = array[left0];
                                array[left0] = t;
                                pivot = left0;
                            }
                            left = left1 = left0 + 1;
                            right = right1 = right0;

                            ta = array[pivot];
                            for(;;) {
                                while( left <= right && !cmp(ta, array[left]) ) {
                                    if( !cmp(array[left], ta) ) {
                                        if( left > left1 ) {
                                            t = array[left1];
                                            array[left1] = array[left];
                                            array[left] = t;
                                        }
                                        swap_cnt = 1;
                                        left1++;
                                    }
                                    left++;
                                }

                                while( left <= right && !cmp(array[right], ta) ) {
                                    if( !cmp(ta, array[right]) ) {
                                        if( right < right1 ) {
                                            t = array[right1];
                                            array[right1] = array[right];
                                            array[right] = t;
                                        }
                                        swap_cnt = 1;
                                        right1--;
                                    }
                                    right--;
                                }

                                if( left > right ) break;
                                
                                t = array[left];
                                array[left] = array[right];
                                array[right] = t;
                                swap_cnt = 1;
                                left++;
                                right--;
                            }

                            if( swap_cnt == 0 ) {
                                left = left0, right = right0;
                                //goto insert_sort;
                                for( ptr = left + 1; ptr <= right; ptr++ ) {
                                    for( ptr2 = ptr; ptr2 > left && cmp(array[ptr2],array[ptr2-1]); ptr2--) {
                                        t = array[ptr2];
                                        array[ptr2] = array[ptr2-1];
                                        array[ptr2-1] = t;
                                    }
                                }
                                break;
                            }

                            n = Math.min( (left1 - left0), (left - left1) );
                            m = (left-n)|0;
                            for( i = 0; i < n; ++i,++m ) {
                                t = array[left0+i];
                                array[left0+i] = array[m];
                                array[m] = t;
                            }

                            n = Math.min( (right0 - right1), (right1 - right) );
                            m = (right0-n+1)|0;
                            for( i = 0; i < n; ++i,++m ) {
                                t = array[left+i];
                                array[left+i] = array[m];
                                array[m] = t;
                            }
                            n = (left - left1);
                            m = (right1 - right);
                            if( n > 1 ) {
                                if( m > 1 ) {
                                    if( n > m ) {
                                        ++sp;
                                        stack[sp<<1] = left0;
                                        stack[(sp<<1)+1] = left0 + n - 1;
                                        left = right0 - m + 1, right = right0;
                                    } else {
                                        ++sp;
                                        stack[sp<<1] = right0 - m + 1;
                                        stack[(sp<<1)+1] = right0;
                                        left = left0, right = left0 + n - 1;
                                    }
                                } else {
                                    left = left0, right = left0 + n - 1;
                                }
                            }
                            else if( m > 1 )
                                left = right0 - m + 1, right = right0;
                            else
                                break;
                        }
                    }
                }
            },

            median: function(array, low, high) {
                var w;
                var middle=0,ll=0,hh=0,median=(low+high)>>1;
                for (;;) {
                    if (high <= low) return array[median];
                    if (high == (low + 1)) {
                        if (array[low] > array[high]) {
                            w = array[low];
                            array[low] = array[high];
                            array[high] = w;
                        }
                        return array[median];
                    }
                    middle = ((low + high) >> 1);
                    if (array[middle] > array[high]) {
                        w = array[middle];
                        array[middle] = array[high];
                        array[high] = w;
                    }
                    if (array[low] > array[high]) {
                        w = array[low];
                        array[low] = array[high];
                        array[high] = w;
                    }
                    if (array[middle] > array[low]) {
                        w = array[middle];
                        array[middle] = array[low];
                        array[low] = w;
                    }
                    ll = (low + 1);
                    w = array[middle];
                    array[middle] = array[ll];
                    array[ll] = w;
                    hh = high;
                    for (;;) {
                        do ++ll; while (array[low] > array[ll]);
                        do --hh; while (array[hh] > array[low]);
                        if (hh < ll) break;
                        w = array[ll];
                        array[ll] = array[hh];
                        array[hh] = w;
                    }
                    w = array[low];
                    array[low] = array[hh];
                    array[hh] = w;
                    if (hh <= median)
                        low = ll;
                    else if (hh >= median)
                        high = (hh - 1);
                }
                return 0;
            }
        };

    })();

    global.math = math;

})(jsfeat);

/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

(function(global) {
    "use strict";
    //

    var imgproc = (function() {

        var _resample_u8 = function(src, dst, nw, nh) {
            var xofs_count=0;
            var ch=src.channel,w=src.cols,h=src.rows;
            var src_d=src.data,dst_d=dst.data;
            var scale_x = w / nw, scale_y = h / nh;
            var inv_scale_256 = (scale_x * scale_y * 0x10000)|0;
            var dx=0,dy=0,sx=0,sy=0,sx1=0,sx2=0,i=0,k=0,fsx1=0.0,fsx2=0.0;
            var a=0,b=0,dxn=0,alpha=0,beta=0,beta1=0;

            var buf_node = jsfeat.cache.get_buffer((nw*ch)<<2);
            var sum_node = jsfeat.cache.get_buffer((nw*ch)<<2);
            var xofs_node = jsfeat.cache.get_buffer((w*2*3)<<2);

            var buf = buf_node.i32;
            var sum = sum_node.i32;
            var xofs = xofs_node.i32;

            for (; dx < nw; dx++) {
                fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
                sx1 = (fsx1 + 1.0 - 1e-6)|0, sx2 = fsx2|0;
                sx1 = Math.min(sx1, w - 1);
                sx2 = Math.min(sx2, w - 1);

                if(sx1 > fsx1) {
                    xofs[k++] = (dx * ch)|0;
                    xofs[k++] = ((sx1 - 1)*ch)|0; 
                    xofs[k++] = ((sx1 - fsx1) * 0x100)|0;
                    xofs_count++;
                }
                for(sx = sx1; sx < sx2; sx++){
                    xofs_count++;
                    xofs[k++] = (dx * ch)|0;
                    xofs[k++] = (sx * ch)|0;
                    xofs[k++] = 256;
                }
                if(fsx2 - sx2 > 1e-3) {
                    xofs_count++;
                    xofs[k++] = (dx * ch)|0;
                    xofs[k++] = (sx2 * ch)|0;
                    xofs[k++] = ((fsx2 - sx2) * 256)|0;
                }
            }

            for (dx = 0; dx < nw * ch; dx++) {
                buf[dx] = sum[dx] = 0;
            }
            dy = 0;
            for (sy = 0; sy < h; sy++) {
                a = w * sy;
                for (k = 0; k < xofs_count; k++) {
                    dxn = xofs[k*3];
                    sx1 = xofs[k*3+1];
                    alpha = xofs[k*3+2];
                    for (i = 0; i < ch; i++) {
                        buf[dxn + i] += src_d[a+sx1+i] * alpha;
                    }
                }
                if ((dy + 1) * scale_y <= sy + 1 || sy == h - 1) {
                    beta = (Math.max(sy + 1 - (dy + 1) * scale_y, 0.0) * 256)|0;
                    beta1 = 256 - beta;
                    b = nw * dy;
                    if (beta <= 0) {
                        for (dx = 0; dx < nw * ch; dx++) {
                            dst_d[b+dx] = Math.min(Math.max((sum[dx] + buf[dx] * 256) / inv_scale_256, 0), 255);
                            sum[dx] = buf[dx] = 0;
                        }
                    } else {
                        for (dx = 0; dx < nw * ch; dx++) {
                            dst_d[b+dx] = Math.min(Math.max((sum[dx] + buf[dx] * beta1) / inv_scale_256, 0), 255);
                            sum[dx] = buf[dx] * beta;
                            buf[dx] = 0;
                        }
                    }
                    dy++;
                } else {
                    for(dx = 0; dx < nw * ch; dx++) {
                        sum[dx] += buf[dx] * 256;
                        buf[dx] = 0;
                    }
                }
            }

            jsfeat.cache.put_buffer(sum_node);
            jsfeat.cache.put_buffer(buf_node);
            jsfeat.cache.put_buffer(xofs_node);
        }

        var _resample = function(src, dst, nw, nh) {
            var xofs_count=0;
            var ch=src.channel,w=src.cols,h=src.rows;
            var src_d=src.data,dst_d=dst.data;
            var scale_x = w / nw, scale_y = h / nh;
            var scale = 1.0 / (scale_x * scale_y);
            var dx=0,dy=0,sx=0,sy=0,sx1=0,sx2=0,i=0,k=0,fsx1=0.0,fsx2=0.0;
            var a=0,b=0,dxn=0,alpha=0.0,beta=0.0,beta1=0.0;

            var buf_node = jsfeat.cache.get_buffer((nw*ch)<<2);
            var sum_node = jsfeat.cache.get_buffer((nw*ch)<<2);
            var xofs_node = jsfeat.cache.get_buffer((w*2*3)<<2);

            var buf = buf_node.f32;
            var sum = sum_node.f32;
            var xofs = xofs_node.f32;

            for (; dx < nw; dx++) {
                fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
                sx1 = (fsx1 + 1.0 - 1e-6)|0, sx2 = fsx2|0;
                sx1 = Math.min(sx1, w - 1);
                sx2 = Math.min(sx2, w - 1);

                if(sx1 > fsx1) {
                    xofs_count++;
                    xofs[k++] = ((sx1 - 1)*ch)|0;
                    xofs[k++] = (dx * ch)|0;
                    xofs[k++] = (sx1 - fsx1) * scale;
                }
                for(sx = sx1; sx < sx2; sx++){
                    xofs_count++;
                    xofs[k++] = (sx * ch)|0;
                    xofs[k++] = (dx * ch)|0; 
                    xofs[k++] = scale;
                }
                if(fsx2 - sx2 > 1e-3) {
                    xofs_count++;
                    xofs[k++] = (sx2 * ch)|0;
                    xofs[k++] = (dx * ch)|0;
                    xofs[k++] = (fsx2 - sx2) * scale;
                }
            }

            for (dx = 0; dx < nw * ch; dx++) {
                buf[dx] = sum[dx] = 0;
            }
            dy = 0;
            for (sy = 0; sy < h; sy++) {
                a = w * sy;
                for (k = 0; k < xofs_count; k++) {
                    sx1 = xofs[k*3]|0;
                    dxn = xofs[k*3+1]|0;
                    alpha = xofs[k*3+2];
                    for (i = 0; i < ch; i++) {
                        buf[dxn + i] += src_d[a+sx1+i] * alpha;
                    }
                }
                if ((dy + 1) * scale_y <= sy + 1 || sy == h - 1) {
                    beta = Math.max(sy + 1 - (dy + 1) * scale_y, 0.0);
                    beta1 = 1.0 - beta;
                    b = nw * dy;
                    if (Math.abs(beta) < 1e-3) {
                        for (dx = 0; dx < nw * ch; dx++) {
                            dst_d[b+dx] = sum[dx] + buf[dx];
                            sum[dx] = buf[dx] = 0;
                        }
                    } else {
                        for (dx = 0; dx < nw * ch; dx++) {
                            dst_d[b+dx] = sum[dx] + buf[dx] * beta1;
                            sum[dx] = buf[dx] * beta;
                            buf[dx] = 0;
                        }
                    }
                    dy++;
                } else {
                    for(dx = 0; dx < nw * ch; dx++) {
                        sum[dx] += buf[dx]; 
                        buf[dx] = 0;
                    }
                }
            }
            jsfeat.cache.put_buffer(sum_node);
            jsfeat.cache.put_buffer(buf_node);
            jsfeat.cache.put_buffer(xofs_node);
        }

        var _convol_u8 = function(buf, src_d, dst_d, w, h, filter, kernel_size, half_kernel) {
            var i=0,j=0,k=0,sp=0,dp=0,sum=0,sum1=0,sum2=0,sum3=0,f0=filter[0],fk=0;
            var w2=w<<1,w3=w*3,w4=w<<2;
            // hor pass
            for (; i < h; ++i) { 
                sum = src_d[sp];
                for (j = 0; j < half_kernel; ++j) {
                    buf[j] = sum;
                }
                for (j = 0; j <= w-2; j+=2) {
                    buf[j + half_kernel] = src_d[sp+j];
                    buf[j + half_kernel+1] = src_d[sp+j+1];
                }
                for (; j < w; ++j) {
                    buf[j + half_kernel] = src_d[sp+j];
                }
                sum = src_d[sp+w-1];
                for (j = w; j < half_kernel + w; ++j) {
                    buf[j + half_kernel] = sum;
                }
                for (j = 0; j <= w-4; j+=4) {
                    sum = buf[j] * f0, 
                    sum1 = buf[j+1] * f0,
                    sum2 = buf[j+2] * f0,
                    sum3 = buf[j+3] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        fk = filter[k];
                        sum += buf[k + j] * fk;
                        sum1 += buf[k + j+1] * fk;
                        sum2 += buf[k + j+2] * fk;
                        sum3 += buf[k + j+3] * fk;
                    }
                    dst_d[dp+j] = Math.min(sum >> 8, 255);
                    dst_d[dp+j+1] = Math.min(sum1 >> 8, 255);
                    dst_d[dp+j+2] = Math.min(sum2 >> 8, 255);
                    dst_d[dp+j+3] = Math.min(sum3 >> 8, 255);
                }
                for (; j < w; ++j) {
                    sum = buf[j] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        sum += buf[k + j] * filter[k];
                    }
                    dst_d[dp+j] = Math.min(sum >> 8, 255);
                }
                sp += w;
                dp += w;
            }

            // vert pass
            for (i = 0; i < w; ++i) {
                sum = dst_d[i];
                for (j = 0; j < half_kernel; ++j) {
                    buf[j] = sum;
                }
                k = i;
                for (j = 0; j <= h-2; j+=2, k+=w2) {
                    buf[j+half_kernel] = dst_d[k];
                    buf[j+half_kernel+1] = dst_d[k+w];
                }
                for (; j < h; ++j, k+=w) {
                    buf[j+half_kernel] = dst_d[k];
                }
                sum = dst_d[(h-1)*w + i];
                for (j = h; j < half_kernel + h; ++j) {
                    buf[j + half_kernel] = sum;
                }
                dp = i;
                for (j = 0; j <= h-4; j+=4, dp+=w4) { 
                    sum = buf[j] * f0, 
                    sum1 = buf[j+1] * f0,
                    sum2 = buf[j+2] * f0,
                    sum3 = buf[j+3] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        fk = filter[k];
                        sum += buf[k + j] * fk;
                        sum1 += buf[k + j+1] * fk;
                        sum2 += buf[k + j+2] * fk;
                        sum3 += buf[k + j+3] * fk;
                    }
                    dst_d[dp] = Math.min(sum >> 8, 255);
                    dst_d[dp+w] = Math.min(sum1 >> 8, 255);
                    dst_d[dp+w2] = Math.min(sum2 >> 8, 255);
                    dst_d[dp+w3] = Math.min(sum3 >> 8, 255);
                }
                for (; j < h; ++j, dp+=w) {
                    sum = buf[j] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        sum += buf[k + j] * filter[k];
                    }
                    dst_d[dp] = Math.min(sum >> 8, 255);
                }
            }
        }

        var _convol = function(buf, src_d, dst_d, w, h, filter, kernel_size, half_kernel) {
            var i=0,j=0,k=0,sp=0,dp=0,sum=0.0,sum1=0.0,sum2=0.0,sum3=0.0,f0=filter[0],fk=0.0;
            var w2=w<<1,w3=w*3,w4=w<<2;
            // hor pass
            for (; i < h; ++i) { 
                sum = src_d[sp];
                for (j = 0; j < half_kernel; ++j) {
                    buf[j] = sum;
                }
                for (j = 0; j <= w-2; j+=2) {
                    buf[j + half_kernel] = src_d[sp+j];
                    buf[j + half_kernel+1] = src_d[sp+j+1];
                }
                for (; j < w; ++j) {
                    buf[j + half_kernel] = src_d[sp+j];
                }
                sum = src_d[sp+w-1];
                for (j = w; j < half_kernel + w; ++j) {
                    buf[j + half_kernel] = sum;
                }
                for (j = 0; j <= w-4; j+=4) {
                    sum = buf[j] * f0, 
                    sum1 = buf[j+1] * f0,
                    sum2 = buf[j+2] * f0,
                    sum3 = buf[j+3] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        fk = filter[k];
                        sum += buf[k + j] * fk;
                        sum1 += buf[k + j+1] * fk;
                        sum2 += buf[k + j+2] * fk;
                        sum3 += buf[k + j+3] * fk;
                    }
                    dst_d[dp+j] = sum;
                    dst_d[dp+j+1] = sum1;
                    dst_d[dp+j+2] = sum2;
                    dst_d[dp+j+3] = sum3;
                }
                for (; j < w; ++j) {
                    sum = buf[j] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        sum += buf[k + j] * filter[k];
                    }
                    dst_d[dp+j] = sum;
                }
                sp += w;
                dp += w;
            }

            // vert pass
            for (i = 0; i < w; ++i) {
                sum = dst_d[i];
                for (j = 0; j < half_kernel; ++j) {
                    buf[j] = sum;
                }
                k = i;
                for (j = 0; j <= h-2; j+=2, k+=w2) {
                    buf[j+half_kernel] = dst_d[k];
                    buf[j+half_kernel+1] = dst_d[k+w];
                }
                for (; j < h; ++j, k+=w) {
                    buf[j+half_kernel] = dst_d[k];
                }
                sum = dst_d[(h-1)*w + i];
                for (j = h; j < half_kernel + h; ++j) {
                    buf[j + half_kernel] = sum;
                }
                dp = i;
                for (j = 0; j <= h-4; j+=4, dp+=w4) { 
                    sum = buf[j] * f0, 
                    sum1 = buf[j+1] * f0,
                    sum2 = buf[j+2] * f0,
                    sum3 = buf[j+3] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        fk = filter[k];
                        sum += buf[k + j] * fk;
                        sum1 += buf[k + j+1] * fk;
                        sum2 += buf[k + j+2] * fk;
                        sum3 += buf[k + j+3] * fk;
                    }
                    dst_d[dp] = sum;
                    dst_d[dp+w] = sum1;
                    dst_d[dp+w2] = sum2;
                    dst_d[dp+w3] = sum3;
                }
                for (; j < h; ++j, dp+=w) {
                    sum = buf[j] * f0;
                    for (k = 1; k < kernel_size; ++k) {
                        sum += buf[k + j] * filter[k];
                    }
                    dst_d[dp] = sum;
                }
            }
        }

        return {
            // TODO: add support for RGB/BGR order
            // for raw arrays
            grayscale: function(src, dst) {
                var srcLength = src.length|0, srcLength_16 = (srcLength - 16)|0;
                var j = 0;
                var coeff_r = 4899, coeff_g = 9617, coeff_b = 1868;

                for (var i = 0; i <= srcLength_16; i += 16, j += 4) {
                    dst[j]     = (src[i] * coeff_r + src[i+1] * coeff_g + src[i+2] * coeff_b + 8192) >> 14;
                    dst[j + 1] = (src[i+4] * coeff_r + src[i+5] * coeff_g + src[i+6] * coeff_b + 8192) >> 14;
                    dst[j + 2] = (src[i+8] * coeff_r + src[i+9] * coeff_g + src[i+10] * coeff_b + 8192) >> 14;
                    dst[j + 3] = (src[i+12] * coeff_r + src[i+13] * coeff_g + src[i+14] * coeff_b + 8192) >> 14;
                }
                for (; i < srcLength; i += 4, ++j) {
                    dst[j] = (src[i] * coeff_r + src[i+1] * coeff_g + src[i+2] * coeff_b + 8192) >> 14;
                }
            },
            // derived from CCV library
            resample: function(src, dst, nw, nh) {
                var h=src.rows,w=src.cols;
                if (h > nh && w > nw) {
                    // using the fast alternative (fix point scale, 0x100 to avoid overflow)
                    if (src.type&jsfeat.U8_t && dst.type&jsfeat.U8_t && h * w / (nh * nw) < 0x100) {
                        _resample_u8(src, dst, nw, nh);
                    } else {
                        _resample(src, dst, nw, nh);
                    }
                }
            },

            box_blur_gray: function(src, dst, radius, options) {
                if (typeof options === "undefined") { options = 0; }
                var w=src.cols, h=src.rows, h2=h<<1, w2=w<<1;
                var i=0,x=0,y=0,end=0;
                var windowSize = ((radius << 1) + 1)|0;
                var radiusPlusOne = (radius + 1)|0, radiusPlus2 = (radiusPlusOne+1)|0;
                var scale = options&jsfeat.BOX_BLUR_NOSCALE ? 1 : (1.0 / (windowSize*windowSize));

                var tmp_buff = jsfeat.cache.get_buffer((w*h)<<2);

                var sum=0, dstIndex=0, srcIndex = 0, nextPixelIndex=0, previousPixelIndex=0;
                var data_i32 = tmp_buff.i32; // to prevent overflow
                var data_u8 = src.data;
                var hold=0;

                // first pass
                // no need to scale 
                //data_u8 = src.data;
                //data_i32 = tmp;
                for (y = 0; y < h; ++y) {
                    dstIndex = y;
                    sum = radiusPlusOne * data_u8[srcIndex];

                    for(i = (srcIndex+1)|0, end=(srcIndex+radius)|0; i <= end; ++i) {
                        sum += data_u8[i];
                    }

                    nextPixelIndex = (srcIndex + radiusPlusOne)|0;
                    previousPixelIndex = srcIndex;
                    hold = data_u8[previousPixelIndex];
                    for(x = 0; x < radius; ++x, dstIndex += h) {
                        data_i32[dstIndex] = sum;
                        sum += data_u8[nextPixelIndex]- hold;
                        nextPixelIndex ++;
                    }
                    for(; x < w-radiusPlus2; x+=2, dstIndex += h2) {
                        data_i32[dstIndex] = sum;
                        sum += data_u8[nextPixelIndex]- data_u8[previousPixelIndex];

                        data_i32[dstIndex+h] = sum;
                        sum += data_u8[nextPixelIndex+1]- data_u8[previousPixelIndex+1];

                        nextPixelIndex +=2;
                        previousPixelIndex +=2;
                    }
                    for(; x < w-radiusPlusOne; ++x, dstIndex += h) {
                        data_i32[dstIndex] = sum;
                        sum += data_u8[nextPixelIndex]- data_u8[previousPixelIndex];

                        nextPixelIndex ++;
                        previousPixelIndex ++;
                    }
                    
                    hold = data_u8[nextPixelIndex-1];
                    for(; x < w; ++x, dstIndex += h) {
                        data_i32[dstIndex] = sum;

                        sum += hold- data_u8[previousPixelIndex];
                        previousPixelIndex ++;
                    }

                    srcIndex += w;
                }
                //
                // second pass
                srcIndex = 0;
                //data_i32 = tmp; // this is a transpose
                data_u8 = dst.data;

                // dont scale result
                if(scale == 1) {
                    for (y = 0; y < w; ++y) {
                        dstIndex = y;
                        sum = radiusPlusOne * data_i32[srcIndex];

                        for(i = (srcIndex+1)|0, end=(srcIndex+radius)|0; i <= end; ++i) {
                            sum += data_i32[i];
                        }

                        nextPixelIndex = srcIndex + radiusPlusOne;
                        previousPixelIndex = srcIndex;
                        hold = data_i32[previousPixelIndex];

                        for(x = 0; x < radius; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum;
                            sum += data_i32[nextPixelIndex]- hold;
                            nextPixelIndex ++;
                        }
                        for(; x < h-radiusPlus2; x+=2, dstIndex += w2) {
                            data_u8[dstIndex] = sum;
                            sum += data_i32[nextPixelIndex]- data_i32[previousPixelIndex];

                            data_u8[dstIndex+w] = sum;
                            sum += data_i32[nextPixelIndex+1]- data_i32[previousPixelIndex+1];

                            nextPixelIndex +=2;
                            previousPixelIndex +=2;
                        }
                        for(; x < h-radiusPlusOne; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum;

                            sum += data_i32[nextPixelIndex]- data_i32[previousPixelIndex];
                            nextPixelIndex ++;
                            previousPixelIndex ++;
                        }
                        hold = data_i32[nextPixelIndex-1];
                        for(; x < h; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum;

                            sum += hold- data_i32[previousPixelIndex];
                            previousPixelIndex ++;
                        }

                        srcIndex += h;
                    }
                } else {
                    for (y = 0; y < w; ++y) {
                        dstIndex = y;
                        sum = radiusPlusOne * data_i32[srcIndex];

                        for(i = (srcIndex+1)|0, end=(srcIndex+radius)|0; i <= end; ++i) {
                            sum += data_i32[i];
                        }

                        nextPixelIndex = srcIndex + radiusPlusOne;
                        previousPixelIndex = srcIndex;
                        hold = data_i32[previousPixelIndex];

                        for(x = 0; x < radius; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum*scale;
                            sum += data_i32[nextPixelIndex]- hold;
                            nextPixelIndex ++;
                        }
                        for(; x < h-radiusPlus2; x+=2, dstIndex += w2) {
                            data_u8[dstIndex] = sum*scale;
                            sum += data_i32[nextPixelIndex]- data_i32[previousPixelIndex];

                            data_u8[dstIndex+w] = sum*scale;
                            sum += data_i32[nextPixelIndex+1]- data_i32[previousPixelIndex+1];

                            nextPixelIndex +=2;
                            previousPixelIndex +=2;
                        }
                        for(; x < h-radiusPlusOne; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum*scale;

                            sum += data_i32[nextPixelIndex]- data_i32[previousPixelIndex];
                            nextPixelIndex ++;
                            previousPixelIndex ++;
                        }
                        hold = data_i32[nextPixelIndex-1];
                        for(; x < h; ++x, dstIndex += w) {
                            data_u8[dstIndex] = sum*scale;

                            sum += hold- data_i32[previousPixelIndex];
                            previousPixelIndex ++;
                        }

                        srcIndex += h;
                    }
                }

                jsfeat.cache.put_buffer(tmp_buff);
            },

            gaussian_blur: function(src, dst, kernel_size, sigma) {
                if (typeof sigma === "undefined") { sigma = 0.0; }
                if (typeof kernel_size === "undefined") { kernel_size = 0; }
                kernel_size = kernel_size == 0 ? (Math.max(1, (4.0 * sigma + 1.0 - 1e-8)) * 2 + 1)|0 : kernel_size;
                var half_kernel = kernel_size >> 1;
                var w = src.cols, h = src.rows;
                var data_type = src.type, is_u8 = data_type&jsfeat.U8_t;
                var src_d = src.data, dst_d = dst.data;
                var buf,filter,buf_sz=(kernel_size + Math.max(h, w))|0;

                var buf_node = jsfeat.cache.get_buffer(buf_sz<<2);
                var filt_node = jsfeat.cache.get_buffer(kernel_size<<2);

                if(is_u8) {
                    buf = buf_node.i32;
                    filter = filt_node.i32;
                } else if(data_type&jsfeat.S32_t) {
                    buf = buf_node.i32;
                    filter = filt_node.f32;
                } else {
                    buf = buf_node.f32;
                    filter = filt_node.f32;
                }

                jsfeat.math.get_gaussian_kernel(kernel_size, sigma, filter, data_type);

                if(is_u8) {
                    _convol_u8(buf, src_d, dst_d, w, h, filter, kernel_size, half_kernel);
                } else {
                    _convol(buf, src_d, dst_d, w, h, filter, kernel_size, half_kernel);
                }

                jsfeat.cache.put_buffer(buf_node);
                jsfeat.cache.put_buffer(filt_node);
            },
            // assume we always need it for u8 image
            pyrdown: function(src, dst, sx, sy) {
                // this is needed for bbf
                if (typeof sx === "undefined") { sx = 0; }
                if (typeof sy === "undefined") { sy = 0; }

                var w = src.cols, h = src.rows;
                var w2 = w >> 1, h2 = h >> 1;
                var _w2 = w2 - (sx << 1), _h2 = h2 - (sy << 1);
                var x=0,y=0,sptr=sx+sy*w,sline=0,dptr=0,dline=0;
                var src_d = src.data, dst_d = dst.data;

                for(y = 0; y < _h2; ++y) {
                    sline = sptr;
                    dline = dptr;
                    for(x = 0; x <= _w2-2; x+=2, dline+=2, sline += 4) {
                        dst_d[dline] = (src_d[sline] + src_d[sline+1] +
                                            src_d[sline+w] + src_d[sline+w+1] + 2) >> 2;
                        dst_d[dline+1] = (src_d[sline+2] + src_d[sline+3] +
                                            src_d[sline+w+2] + src_d[sline+w+3] + 2) >> 2;
                    }
                    for(; x < _w2; ++x, ++dline, sline += 2) {
                        dst_d[dline] = (src_d[sline] + src_d[sline+1] +
                                            src_d[sline+w] + src_d[sline+w+1] + 2) >> 2;
                    }
                    sptr += w << 1;
                    dptr += w2;
                }
            },

            // dst: [gx,gy,...]
            scharr_derivatives: function(src, dst) {
                var w = src.cols, h = src.rows;
                var dstep = w<<1,x=0,y=0,x1=0,a,b,c,d,e,f;
                var srow0=0,srow1=0,srow2=0,drow=0;
                var trow0,trow1;
                var img = src.data, gxgy=dst.data;

                var buf0_node = jsfeat.cache.get_buffer((w+2)<<2);
                var buf1_node = jsfeat.cache.get_buffer((w+2)<<2);

                if(src.type&jsfeat.U8_t || src.type&jsfeat.S32_t) {
                    trow0 = buf0_node.i32;
                    trow1 = buf1_node.i32;
                } else {
                    trow0 = buf0_node.f32;
                    trow1 = buf1_node.f32;
                }

                for(; y < h; ++y, srow1+=w) {
                    srow0 = ((y > 0 ? y-1 : 1)*w)|0;
                    srow2 = ((y < h-1 ? y+1 : h-2)*w)|0;
                    drow = (y*dstep)|0;
                    // do vertical convolution
                    for(x = 0, x1 = 1; x <= w-2; x+=2, x1+=2) {
                        a = img[srow0+x], b = img[srow2+x];
                        trow0[x1] = ( (a + b)*3 + (img[srow1+x])*10 );
                        trow1[x1] = ( b - a );
                        //
                        a = img[srow0+x+1], b = img[srow2+x+1];
                        trow0[x1+1] = ( (a + b)*3 + (img[srow1+x+1])*10 );
                        trow1[x1+1] = ( b - a );
                    }
                    for(; x < w; ++x, ++x1) {
                        a = img[srow0+x], b = img[srow2+x];
                        trow0[x1] = ( (a + b)*3 + (img[srow1+x])*10 );
                        trow1[x1] = ( b - a );
                    }
                    // make border
                    x = (w + 1)|0;
                    trow0[0] = trow0[1]; trow0[x] = trow0[w];
                    trow1[0] = trow1[1]; trow1[x] = trow1[w];
                    // do horizontal convolution, interleave the results and store them
                    for(x = 0; x <= w-4; x+=4) {
                        a = trow1[x+2], b = trow1[x+1], c = trow1[x+3], d = trow1[x+4],
                        e = trow0[x+2], f = trow0[x+3];
                        gxgy[drow++] = ( e - trow0[x] );
                        gxgy[drow++] = ( (a + trow1[x])*3 + b*10 );
                        gxgy[drow++] = ( f - trow0[x+1] );
                        gxgy[drow++] = ( (c + b)*3 + a*10 );

                        gxgy[drow++] = ( (trow0[x+4] - e) );
                        gxgy[drow++] = ( ((d + a)*3 + c*10) );
                        gxgy[drow++] = ( (trow0[x+5] - f) );
                        gxgy[drow++] = ( ((trow1[x+5] + c)*3 + d*10) );
                    }
                    for(; x < w; ++x) {
                        gxgy[drow++] = ( (trow0[x+2] - trow0[x]) );
                        gxgy[drow++] = ( ((trow1[x+2] + trow1[x])*3 + trow1[x+1]*10) );
                    }
                }
                jsfeat.cache.put_buffer(buf0_node);
                jsfeat.cache.put_buffer(buf1_node);
            },

            // compute gradient using Sobel kernel [1 2 1] * [-1 0 1]^T
            // dst: [gx,gy,...]
            sobel_derivatives: function(src, dst) {
                var w = src.cols, h = src.rows;
                var dstep = w<<1,x=0,y=0,x1=0,a,b,c,d,e,f;
                var srow0=0,srow1=0,srow2=0,drow=0;
                var trow0,trow1;
                var img = src.data, gxgy=dst.data;

                var buf0_node = jsfeat.cache.get_buffer((w+2)<<2);
                var buf1_node = jsfeat.cache.get_buffer((w+2)<<2);

                if(src.type&jsfeat.U8_t || src.type&jsfeat.S32_t) {
                    trow0 = buf0_node.i32;
                    trow1 = buf1_node.i32;
                } else {
                    trow0 = buf0_node.f32;
                    trow1 = buf1_node.f32;
                }

                for(; y < h; ++y, srow1+=w) {
                    srow0 = ((y > 0 ? y-1 : 1)*w)|0;
                    srow2 = ((y < h-1 ? y+1 : h-2)*w)|0;
                    drow = (y*dstep)|0;
                    // do vertical convolution
                    for(x = 0, x1 = 1; x <= w-2; x+=2, x1+=2) {
                        a = img[srow0+x], b = img[srow2+x];
                        trow0[x1] = ( (a + b) + (img[srow1+x]*2) );
                        trow1[x1] = ( b - a );
                        //
                        a = img[srow0+x+1], b = img[srow2+x+1];
                        trow0[x1+1] = ( (a + b) + (img[srow1+x+1]*2) );
                        trow1[x1+1] = ( b - a );
                    }
                    for(; x < w; ++x, ++x1) {
                        a = img[srow0+x], b = img[srow2+x];
                        trow0[x1] = ( (a + b) + (img[srow1+x]*2) );
                        trow1[x1] = ( b - a );
                    }
                    // make border
                    x = (w + 1)|0;
                    trow0[0] = trow0[1]; trow0[x] = trow0[w];
                    trow1[0] = trow1[1]; trow1[x] = trow1[w];
                    // do horizontal convolution, interleave the results and store them
                    for(x = 0; x <= w-4; x+=4) {
                        a = trow1[x+2], b = trow1[x+1], c = trow1[x+3], d = trow1[x+4],
                        e = trow0[x+2], f = trow0[x+3];
                        gxgy[drow++] = ( e - trow0[x] );
                        gxgy[drow++] = ( a + trow1[x] + b*2 );
                        gxgy[drow++] = ( f - trow0[x+1] );
                        gxgy[drow++] = ( c + b + a*2 );

                        gxgy[drow++] = ( trow0[x+4] - e );
                        gxgy[drow++] = ( d + a + c*2 );
                        gxgy[drow++] = ( trow0[x+5] - f );
                        gxgy[drow++] = ( trow1[x+5] + c + d*2 );
                    }
                    for(; x < w; ++x) {
                        gxgy[drow++] = ( trow0[x+2] - trow0[x] );
                        gxgy[drow++] = ( trow1[x+2] + trow1[x] + trow1[x+1]*2 );
                    }
                }
                jsfeat.cache.put_buffer(buf0_node);
                jsfeat.cache.put_buffer(buf1_node);
            },

            // please note: 
            // dst_(type) size should be cols = src.cols+1, rows = src.rows+1
            compute_integral_image: function(src, dst_sum, dst_sqsum, dst_tilted) {
                var w0=src.cols|0,h0=src.rows|0,src_d=src.data;
                var w1=(w0+1)|0;
                var s=0,s2=0,p=0,pup=0,i=0,j=0,v=0,k=0;

                if(dst_sum && dst_sqsum) {
                    // fill first row with zeros
                    for(; i < w1; ++i) {
                        dst_sum[i] = 0, dst_sqsum[i] = 0;
                    }
                    p = (w1+1)|0, pup = 1;
                    for(i = 0, k = 0; i < h0; ++i, ++p, ++pup) {
                        s = s2 = 0;
                        for(j = 0; j <= w0-2; j+=2, k+=2, p+=2, pup+=2) {
                            v = src_d[k];
                            s += v, s2 += v*v;
                            dst_sum[p] = dst_sum[pup] + s;
                            dst_sqsum[p] = dst_sqsum[pup] + s2;

                            v = src_d[k+1];
                            s += v, s2 += v*v;
                            dst_sum[p+1] = dst_sum[pup+1] + s;
                            dst_sqsum[p+1] = dst_sqsum[pup+1] + s2;
                        }
                        for(; j < w0; ++j, ++k, ++p, ++pup) {
                            v = src_d[k];
                            s += v, s2 += v*v;
                            dst_sum[p] = dst_sum[pup] + s;
                            dst_sqsum[p] = dst_sqsum[pup] + s2;
                        }
                    }
                } else if(dst_sum) {
                    // fill first row with zeros
                    for(; i < w1; ++i) {
                        dst_sum[i] = 0;
                    }
                    p = (w1+1)|0, pup = 1;
                    for(i = 0, k = 0; i < h0; ++i, ++p, ++pup) {
                        s = 0;
                        for(j = 0; j <= w0-2; j+=2, k+=2, p+=2, pup+=2) {
                            s += src_d[k];
                            dst_sum[p] = dst_sum[pup] + s;
                            s += src_d[k+1];
                            dst_sum[p+1] = dst_sum[pup+1] + s;
                        }
                        for(; j < w0; ++j, ++k, ++p, ++pup) {
                            s += src_d[k];
                            dst_sum[p] = dst_sum[pup] + s;
                        }
                    }
                } else if(dst_sqsum) {
                    // fill first row with zeros
                    for(; i < w1; ++i) {
                        dst_sqsum[i] = 0;
                    }
                    p = (w1+1)|0, pup = 1;
                    for(i = 0, k = 0; i < h0; ++i, ++p, ++pup) {
                        s2 = 0;
                        for(j = 0; j <= w0-2; j+=2, k+=2, p+=2, pup+=2) {
                            v = src_d[k];
                            s2 += v*v;
                            dst_sqsum[p] = dst_sqsum[pup] + s2;
                            v = src_d[k+1];
                            s2 += v*v;
                            dst_sqsum[p+1] = dst_sqsum[pup+1] + s2;
                        }
                        for(; j < w0; ++j, ++k, ++p, ++pup) {
                            v = src_d[k];
                            s2 += v*v;
                            dst_sqsum[p] = dst_sqsum[pup] + s2;
                        }
                    }
                }

                if(dst_tilted) {
                    // fill first row with zeros
                    for(i = 0; i < w1; ++i) {
                        dst_tilted[i] = 0;
                    }
                    // diagonal
                    p = (w1+1)|0, pup = 0;
                    for(i = 0, k = 0; i < h0; ++i, ++p, ++pup) {
                        for(j = 0; j <= w0-2; j+=2, k+=2, p+=2, pup+=2) {
                            dst_tilted[p] = src_d[k] + dst_tilted[pup];
                            dst_tilted[p+1] = src_d[k+1] + dst_tilted[pup+1];
                        }
                        for(; j < w0; ++j, ++k, ++p, ++pup) {
                            dst_tilted[p] = src_d[k] + dst_tilted[pup];
                        }
                    }
                    // diagonal
                    p = (w1+w0)|0, pup = w0;
                    for(i = 0; i < h0; ++i, p+=w1, pup+=w1) {
                        dst_tilted[p] += dst_tilted[pup];
                    }

                    for(j = w0-1; j > 0; --j) {
                        p = j+h0*w1, pup=p-w1;
                        for(i = h0; i > 0; --i, p-=w1, pup-=w1) {
                            dst_tilted[p] += dst_tilted[pup] + dst_tilted[pup+1];
                        }
                    }
                }
            },
            equalize_histogram: function(src, dst) {
                var w=src.cols,h=src.rows,src_d=src.data,dst_d=dst.data,size=w*h;
                var i=0,prev=0,hist0,norm;

                var hist0_node = jsfeat.cache.get_buffer(256<<2);
                hist0 = hist0_node.i32;
                for(; i < 256; ++i) hist0[i] = 0;
                for (i = 0; i < size; ++i) {
                    ++hist0[src_d[i]];
                }

                prev = hist0[0];
                for (i = 1; i < 256; ++i) {
                    prev = hist0[i] += prev;
                }

                norm = 255 / size;
                for (i = 0; i < size; ++i) {
                    dst_d[i] = (hist0[src_d[i]] * norm + 0.5)|0;
                }
                jsfeat.cache.put_buffer(hist0_node);
            },

            canny: function(src, dst, low_thresh, high_thresh) {
                var w=src.cols,h=src.rows,src_d=src.data,dst_d=dst.data;
                var i=0,j=0,grad=0,w2=w<<1,_grad=0,suppress=0,f=0,x=0,y=0,s=0;
                var tg22x=0,tg67x=0;

                // cache buffers
                var dxdy_node = jsfeat.cache.get_buffer((h * w2)<<2);
                var buf_node = jsfeat.cache.get_buffer((3 * (w + 2))<<2);
                var map_node = jsfeat.cache.get_buffer(((h+2) * (w + 2))<<2);
                var stack_node = jsfeat.cache.get_buffer((h * w)<<2);
                

                var buf = buf_node.i32;
                var map = map_node.i32;
                var stack = stack_node.i32;
                var dxdy = dxdy_node.i32;
                var dxdy_m = new jsfeat.matrix_t(w, h, jsfeat.S32C2_t, dxdy_node.data);
                var row0=1,row1=(w+2+1)|0,row2=(2*(w+2)+1)|0,map_w=(w+2)|0,map_i=(map_w+1)|0,stack_i=0;

                this.sobel_derivatives(src, dxdy_m);

                if(low_thresh > high_thresh) {
                    i = low_thresh;
                    low_thresh = high_thresh;
                    high_thresh = i;
                }

                i = (3 * (w + 2))|0;
                while(--i>=0) {
                    buf[i] = 0;
                }

                i = ((h+2) * (w + 2))|0;
                while(--i>=0) {
                    map[i] = 0;
                }

                for (; j < w; ++j, grad+=2) {
                    //buf[row1+j] = Math.abs(dxdy[grad]) + Math.abs(dxdy[grad+1]);
                    x = dxdy[grad], y = dxdy[grad+1];
                    //buf[row1+j] = x*x + y*y;
                    buf[row1+j] = ((x ^ (x >> 31)) - (x >> 31)) + ((y ^ (y >> 31)) - (y >> 31));
                }

                for(i=1; i <= h; ++i, grad+=w2) {
                    if(i == h) {
                        j = row2+w;
                        while(--j>=row2) {
                            buf[j] = 0;
                        }
                    } else {
                        for (j = 0; j < w; j++) {
                            //buf[row2+j] =  Math.abs(dxdy[grad+(j<<1)]) + Math.abs(dxdy[grad+(j<<1)+1]);
                            x = dxdy[grad+(j<<1)], y = dxdy[grad+(j<<1)+1];
                            //buf[row2+j] = x*x + y*y;
                            buf[row2+j] = ((x ^ (x >> 31)) - (x >> 31)) + ((y ^ (y >> 31)) - (y >> 31));
                        }
                    }
                    _grad = (grad - w2)|0;
                    map[map_i-1] = 0;
                    suppress = 0;
                    for(j = 0; j < w; ++j, _grad+=2) {
                        f = buf[row1+j];
                        if (f > low_thresh) {
                            x = dxdy[_grad];
                            y = dxdy[_grad+1];
                            s = x ^ y;
                            // seems ot be faster than Math.abs
                            x = ((x ^ (x >> 31)) - (x >> 31))|0;
                            y = ((y ^ (y >> 31)) - (y >> 31))|0;
                            //x * tan(22.5) x * tan(67.5) == 2 * x + x * tan(22.5)
                            tg22x = x * 13573;
                            tg67x = tg22x + ((x + x) << 15);
                            y <<= 15;
                            if (y < tg22x) {
                                if (f > buf[row1+j-1] && f >= buf[row1+j+1]) {
                                    if (f > high_thresh && !suppress && map[map_i+j-map_w] != 2) {
                                        map[map_i+j] = 2;
                                        suppress = 1;
                                        stack[stack_i++] = map_i + j;
                                    } else {
                                        map[map_i+j] = 1;
                                    }
                                    continue;
                                }
                            } else if (y > tg67x) {
                                if (f > buf[row0+j] && f >= buf[row2+j]) {
                                    if (f > high_thresh && !suppress && map[map_i+j-map_w] != 2) {
                                        map[map_i+j] = 2;
                                        suppress = 1;
                                        stack[stack_i++] = map_i + j;
                                    } else {
                                        map[map_i+j] = 1;
                                    }
                                    continue;
                                }
                            } else {
                                s = s < 0 ? -1 : 1;
                                if (f > buf[row0+j-s] && f > buf[row2+j+s]) {
                                    if (f > high_thresh && !suppress && map[map_i+j-map_w] != 2) {
                                        map[map_i+j] = 2;
                                        suppress = 1;
                                        stack[stack_i++] = map_i + j;
                                    } else {
                                        map[map_i+j] = 1;
                                    }
                                    continue;
                                }
                            }
                        }
                        map[map_i+j] = 0;
                        suppress = 0;
                    }
                    map[map_i+w] = 0;
                    map_i += map_w;
                    j = row0;
                    row0 = row1;
                    row1 = row2;
                    row2 = j;
                }

                j = map_i - map_w - 1;
                for(i = 0; i < map_w; ++i, ++j) {
                    map[j] = 0;
                }
                // path following
                while(stack_i > 0) {
                    map_i = stack[--stack_i];
                    map_i -= map_w+1;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += 1;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += 1;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += map_w;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i -= 2;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += map_w;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += 1;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                    map_i += 1;
                    if(map[map_i] == 1) map[map_i] = 2, stack[stack_i++] = map_i;
                }

                map_i = map_w + 1;
                row0 = 0;
                for(i = 0; i < h; ++i, map_i+=map_w) {
                    for(j = 0; j < w; ++j) {
                        dst_d[row0++] = (map[map_i+j] == 2) * 0xff;
                    }
                }

                // free buffers
                jsfeat.cache.put_buffer(dxdy_node);
                jsfeat.cache.put_buffer(buf_node);
                jsfeat.cache.put_buffer(map_node);
                jsfeat.cache.put_buffer(stack_node);
            },
            // transform is 3x3 matrix_t
            warp_perspective: function(src, dst, transform, fill_value) {
                if (typeof fill_value === "undefined") { fill_value = 0; }
                var src_width=src.cols|0, src_height=src.rows|0, dst_width=dst.cols|0, dst_height=dst.rows|0;
                var src_d=src.data, dst_d=dst.data;
                var x=0,y=0,off=0,ixs=0,iys=0,xs=0.0,ys=0.0,xs0=0.0,ys0=0.0,ws=0.0,sc=0.0,a=0.0,b=0.0,p0=0.0,p1=0.0;
                var td=transform.data;
                var m00=td[0],m01=td[1],m02=td[2],
                    m10=td[3],m11=td[4],m12=td[5],
                    m20=td[6],m21=td[7],m22=td[8];

                for(var dptr = 0; y < dst_height; ++y) {
                    xs0 = m01 * y + m02,
                    ys0 = m11 * y + m12,
                    ws  = m21 * y + m22;
                    for(x = 0; x < dst_width; ++x, ++dptr, xs0+=m00, ys0+=m10, ws+=m20) {
                        sc = 1.0 / ws;
                        xs = xs0 * sc, ys = ys0 * sc;
                        ixs = xs | 0, iys = ys | 0;

                        if(xs > 0 && ys > 0 && ixs < (src_width - 1) && iys < (src_height - 1)) {
                            a = Math.max(xs - ixs, 0.0);
                            b = Math.max(ys - iys, 0.0);
                            off = (src_width*iys + ixs)|0;

                            p0 = src_d[off] +  a * (src_d[off+1] - src_d[off]);
                            p1 = src_d[off+src_width] + a * (src_d[off+src_width+1] - src_d[off+src_width]);

                            dst_d[dptr] = p0 + b * (p1 - p0);
                        }
                        else dst_d[dptr] = fill_value;
                    }
                }
            },
            // transform is 3x3 or 2x3 matrix_t only first 6 values referenced
            warp_affine: function(src, dst, transform, fill_value) {
                if (typeof fill_value === "undefined") { fill_value = 0; }
                var src_width=src.cols, src_height=src.rows, dst_width=dst.cols, dst_height=dst.rows;
                var src_d=src.data, dst_d=dst.data;
                var x=0,y=0,off=0,ixs=0,iys=0,xs=0.0,ys=0.0,a=0.0,b=0.0,p0=0.0,p1=0.0;
                var td=transform.data;
                var m00=td[0],m01=td[1],m02=td[2],
                    m10=td[3],m11=td[4],m12=td[5];

                for(var dptr = 0; y < dst_height; ++y) {
                    xs = m01 * y + m02;
                    ys = m11 * y + m12;
                    for(x = 0; x < dst_width; ++x, ++dptr, xs+=m00, ys+=m10) {
                        ixs = xs | 0; iys = ys | 0;

                        if(xs > 0 && ys > 0 && ixs < (src_width - 1) && iys < (src_height - 1)) {
                            a = Math.max(xs - ixs, 0.0);
                            b = Math.max(ys - iys, 0.0);
                            off = src_width*iys + ixs;

                            p0 = src_d[off] +  a * (src_d[off+1] - src_d[off]);
                            p1 = src_d[off+src_width] + a * (src_d[off+src_width+1] - src_d[off+src_width]);

                            dst_d[dptr] = p0 + b * (p1 - p0);
                        }
                        else dst_d[dptr] = fill_value;
                    }
                }
            }
        };
    })();

    global.imgproc = imgproc;

})(jsfeat);



/**
 * @author Eugene Zatepyakin / http://inspirit.ru/
 */

(function(lib) {
    "use strict";

    if (typeof module === "undefined" || typeof module.exports === "undefined") {
        // in a browser, define its namespaces in global
        window.jsfeat = lib;
    } else {
        // in commonjs, or when AMD wrapping has been applied, define its namespaces as exports
        module.exports = lib;
    }
})(jsfeat);