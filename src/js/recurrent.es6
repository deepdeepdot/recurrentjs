import {assert, zeros} from "./helper.es6";
import Mat from "./mat.es6";
import Graph from "./graph.es6";
import Solver from "./solver.es6";

var R = {}; // the Recurrent library

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

var randf = (a, b)=> Math.random()*(b-a)+a;
var randi = (a, b) => Math.floor(Math.random()*(b-a)+a);
var randn = (mu, std) => mu+gaussRandom()*std;

// return Mat but filled with random numbers from gaussian
var RandMat = function(n,d,mu,std) {
  var m = new Mat(n, d);
  //fillRandn(m,mu,std);
  fillRand(m,-std,std); // kind of :P
  return m;
}

// Mat utils
// fill matrix with random gaussian numbers
var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }

var softmax = function(m) {
  var out = new Mat(m.n, m.d); // probability volume
  var maxval = -999999;
  for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

  var s = 0.0;
  for(var i=0,n=m.w.length;i<n;i++) { 
    out.w[i] = Math.exp(m.w[i] - maxval);
    s += out.w[i];
  }
  for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

  // no backward pass here needed
  // since we will use the computed probabilities outside
  // to set gradients directly on m
  return out;
}

var initLSTM = function(input_size, hidden_sizes, output_size) {
  // hidden size should be a list

  var model = {};
  for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
    var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
    var hidden_size = hidden_sizes[d];

    // gates parameters
    model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
    model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
    model['bi'+d] = new Mat(hidden_size, 1);
    model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
    model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
    model['bf'+d] = new Mat(hidden_size, 1);
    model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
    model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
    model['bo'+d] = new Mat(hidden_size, 1);
    // cell write params
    model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
    model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
    model['bc'+d] = new Mat(hidden_size, 1);
  }
  // decoder params
  model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
  model['bd'] = new Mat(output_size, 1);
  return model;
}

var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration

  if(typeof prev.h === 'undefined') {
    var hidden_prevs = [];
    var cell_prevs = [];
    for(var d=0;d<hidden_sizes.length;d++) {
      hidden_prevs.push(new Mat(hidden_sizes[d],1)); 
      cell_prevs.push(new Mat(hidden_sizes[d],1)); 
    }
  } else {
    var hidden_prevs = prev.h;
    var cell_prevs = prev.c;
  }

  var hidden = [];
  var cell = [];
  for(var d=0;d<hidden_sizes.length;d++) {

    var input_vector = d === 0 ? x : hidden[d-1];
    var hidden_prev = hidden_prevs[d];
    var cell_prev = cell_prevs[d];

    // input gate
    var h0 = G.mul(model['Wix'+d], input_vector);
    var h1 = G.mul(model['Wih'+d], hidden_prev);
    var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

    // forget gate
    var h2 = G.mul(model['Wfx'+d], input_vector);
    var h3 = G.mul(model['Wfh'+d], hidden_prev);
    var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

    // output gate
    var h4 = G.mul(model['Wox'+d], input_vector);
    var h5 = G.mul(model['Woh'+d], hidden_prev);
    var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

    // write operation on cells
    var h6 = G.mul(model['Wcx'+d], input_vector);
    var h7 = G.mul(model['Wch'+d], hidden_prev);
    var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

    // compute new cell activation
    var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
    var cell_d = G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations
    var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

    hidden.push(hidden_d);
    cell.push(cell_d);
  }

  // one decoder to outputs at end
  var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

  // return cell memory, hidden representation and output
  return {'h':hidden, 'c':cell, 'o' : output};
}

var initRNN = function(input_size, hidden_sizes, output_size) {
  // hidden size should be a list

  var model = {};
  for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
    var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
    var hidden_size = hidden_sizes[d];
    model['Wxh'+d] = new R.RandMat(hidden_size, prev_size , 0, 0.08);
    model['Whh'+d] = new R.RandMat(hidden_size, hidden_size, 0, 0.08);
    model['bhh'+d] = new Mat(hidden_size, 1);
  }
  // decoder params
  model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
  model['bd'] = new Mat(output_size, 1);
  return model;
}

 var forwardRNN = function(G, model, hidden_sizes, x, prev) {
  // forward prop for a single tick of RNN
  // G is graph to append ops to
  // model contains RNN parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden activations from last step

  if(typeof prev.h === 'undefined') {
    var hidden_prevs = [];
    for(var d=0;d<hidden_sizes.length;d++) {
      hidden_prevs.push(new Mat(hidden_sizes[d],1)); 
    }
  } else {
    var hidden_prevs = prev.h;
  }

  var hidden = [];
  for(var d=0;d<hidden_sizes.length;d++) {

    var input_vector = d === 0 ? x : hidden[d-1];
    var hidden_prev = hidden_prevs[d];

    var h0 = G.mul(model['Wxh'+d], input_vector);
    var h1 = G.mul(model['Whh'+d], hidden_prev);
    var hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh'+d]));

    hidden.push(hidden_d);
  }

  // one decoder to outputs at end
  var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

  // return cell memory, hidden representation and output
  return {'h':hidden, 'o' : output};
}

var maxi = function(w) {
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

var samplei = function(w) {
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

// various utils
R.maxi = maxi;
R.samplei = samplei;
R.randi = randi;
R.softmax = softmax;
R.assert = assert;

// classes
R.RandMat = RandMat;

R.forwardLSTM = forwardLSTM;
R.initLSTM = initLSTM;
R.forwardRNN = forwardRNN;
R.initRNN = initRNN;

export default R;