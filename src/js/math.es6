import {assert} from "./helper.es6";

// Random numbers utils

var randf = (a, b)=> Math.random()*(b-a)+a;
var randi = (a, b) => Math.floor(Math.random()*(b-a)+a);
var randn = (mu, std) => mu+gaussRandom()*std;


let gaussRandom = (() => {
  // polar form of the Box-Muller transformation
  // faster and more robust numerically
  // https://www.taygeta.com/random/gaussian.html

  let cache_value = null;

  return () => {
    if(cache_value!==null) {
      let temp = cache_value
      cache_value = null;
      return temp;
    }
    
    let u, v, r;

    while(!r || r >= 1){
      u = randf(-1, 1);
      v = randf(-1, 1);
      r = u*u + v*v;
    }

    let c = Math.sqrt(-2*Math.log(r)/r);
    cache_value = v*c;
    return u*c;
  }
})()

var fillRandn = (m, mu, std) => { 
  for(var i=0,n=m.w.length;i<n;i++) { 
    m.w[i] = randn(mu, std); 
  }
}

var fillRand = (m, lo, hi) => {
  for(var i=0,n=m.w.length;i<n;i++) {
    m.w[i] = randf(lo, hi); 
  }
}

var samplei = (w) => {
  // sample argmax from w, assuming w are 
  // probabilities that sum to one
  var r = randf(0,1);
  var x = 0.0;
  var i = 0;
  
  while(true) {
    x += w[i];
    if(x > r) { return i; }
    i++;
  }
  
  throw Error("Error sampling");
}

var maxi = (w) => {
  // argmax of array w
  var maxv = w[0];
  var maxix = 0;
  
  for(var i=1,n=w.length;i<n;i++) {
    var v = w[i];
    if(v > maxv) {
      maxix = i;
      maxv = v;
    }
  }
  return maxix;
}

// helper function returns array of zeros of length n
// and uses typed arrays if available
let zeros = (n) => {
  assert(!isNaN(n));

  if(typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    var arr = new Array(n);
    for(var i=0;i<n;i++) { arr[i] = 0; }
    return arr;
  } else {
    return new Float64Array(n);
  }
}

let median = (values) => {
  values.sort((a,b) => a - b);
  var half = Math.floor(values.length/2);
  if(values.length % 2) return values[half];
  else return (values[half-1] + values[half]) / 2.0;
}

// helper function for computing sigmoid
let sig = (x) => 1.0/(1+Math.exp(-x));

export default {
	fillRandn,
	fillRand,
	randf,
	randi,
	randn,
	samplei,
	maxi,
  gaussRandom,
  zeros,
  median,
  sig
}