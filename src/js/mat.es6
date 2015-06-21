import {assert, zeros} from "./helper.es6";

// Mat holds a matrix
class Mat {
  constructor(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }

  get(row, col) { 
    // slow but careful accessor function
    // we want row-major order
    var ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  }

  set(row, col, v) {
    // slow but careful accessor function
    var ix = (this.d * row) + col;
    assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v; 
  }
  
  toJSON() {
    var json = {};
    json['n'] = this.n;
    json['d'] = this.d;
    json['w'] = this.w;
    return json;
  }
  
  fromJSON(json) {
    this.n = json.n;
    this.d = json.d;
    this.w = zeros(this.n * this.d);
    this.dw = zeros(this.n * this.d);
    for(var i=0,n=this.n * this.d;i<n;i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
}

export default Mat