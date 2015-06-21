var randf = (a, b)=> Math.random()*(b-a)+a;
var randi = (a, b) => Math.floor(Math.random()*(b-a)+a);
var randn = (mu, std) => mu+gaussRandom()*std;

// Random numbers utils
var return_v = false;
var v_val = 0.0;

var gaussRandom = () => {
  if(return_v) {
    return_v = false;
    return v_val; 
  }
  var u = 2*Math.random()-1;
  var v = 2*Math.random()-1;
  var r = u*u + v*v;
  if(r == 0 || r > 1) return gaussRandom();
  var c = Math.sqrt(-2*Math.log(r)/r);
  v_val = v*c; // cache this
  return_v = true;
  return u*c;
}

var fillRandn = (m, mu, std) => { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
var fillRand = (m, lo, hi) => { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }

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
  return w.length - 1; // pretty sure we should never get here?
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

export default {
	fillRandn,
	fillRand,
	randf,
	randi,
	randn,
	samplei,
	maxi
}