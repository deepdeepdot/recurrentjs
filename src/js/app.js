import _ from "lodash";
import $ from "jquery";
import work from "webworkify";

import {forwardIndex, forwardLSTM, initLSTM, forwardRNN, initRNN} from "./recurrent";
import Rvis from "./vis.js";
import {RandMat, Mat, softmax} from "./mat.js";
import {median, randi, maxi, samplei, gaussRandom} from "./math";
import Graph from "./graph.js";
import Solver from "./solver.js";
import Ticker from "./ticker";
import WebWorker from "./webworker";

import React from "react";
import {classSet as cx} from "react-addons";
import ChooseInputTextComponent from "./components/choose_input_text_component";
import Slider from "./components/slider";

// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be

// various global var inits
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var data_sents = [];
var solver = new Solver(); // should be class because it needs memory for step caches
var pplGraph;

var generator, hidden_sizes, letter_size, regc, learning_rate, clipval, ch, lh, logprobs, probs, step_cache_out, step_cache;

var predict_num_lines = 10; // number of lines for the prediction to show
var model = {};

var initVocab = (sents, count_threshold) => {
  // go over all characters and keep track of all unique ones seen
  var txt = sents.join(''); // concat all

  // count up all characters
  var d = {};
  for(var i=0,n=txt.length;i<n;i++) {
    var txti = txt[i];
    if(txti in d) { d[txti] += 1; } 
    else { d[txti] = 1; }
  }

  // filter by count threshold and create pointers
  letterToIndex = {};
  indexToLetter = {};
  vocab = [];
  // NOTE: start at one because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  var q = 1; 
  for(ch in d) {
    if(d[ch] >= count_threshold) {
      // add character to vocab
      letterToIndex[ch] = q;
      indexToLetter[q] = ch;
      vocab.push(ch);
      q++;
    }
  }

  // globals written: indexToLetter, letterToIndex, vocab (list), and:
  input_size = vocab.length + 1;
  output_size = vocab.length + 1;
  epoch_size = sents.length;
  $("#prepro_status").text('found ' + vocab.length + ' distinct characters: ' + vocab.sort().join(''));
}

var utilAddToModel = (modelto, modelfrom) => {
  for(var k in modelfrom) {
    modelto[k] = modelfrom[k];
  }
}

var initModel = () => {
  // letter embedding vectors
  let model = {
    "Wil": new RandMat(input_size, letter_size , 0, 0.08)
  };
  
  let fn = (generator==='rnn') ? initRNN : initLSTM;
  let network = fn(letter_size, hidden_sizes, output_size)
  
  _.merge(model, network);
  return model;
}

window.chosen_input_file = window.input_files[randi(0, window.input_files.length)];

var reinit = () => {
  // note: reinit writes global vars
  
  // eval options to set some globals

  eval($("#newnet").val());
  
  render_learning_slider();

  solver = new Solver(); // reinit solver
  pplGraph = pplGraph || new Rvis.Graph("#pplgraph");
  pplGraph.reset();

  ppl_list = [];
  ticker.reset();

  // process the input, filter out blanks

  return Promise.resolve($.get(`/data/${chosen_input_file}`)).then((text) => {
    $('#input_text').text(text);

    var data_sents_raw = text.split('\n');

    data_sents = [];

    for(var i=0;i<data_sents_raw.length;i++) {
      var sent = data_sents_raw[i].trim();
      if(sent.length > 0) {
        data_sents.push(sent);
      }
    }

    initVocab(data_sents, 1); // takes count threshold for characters
    model = initModel();

    render_input_file_container();
  });
}

var saveModel = () => {
  var out = {};
  out['hidden_sizes'] = hidden_sizes;
  out['generator'] = generator;
  out['letter_size'] = letter_size;
  var model_out = {};
  for(var k in model) {
    model_out[k] = model[k].toJSON();
  }
  out['model'] = model_out;
  var solver_out = {};
  solver_out['decay_rate'] = solver.decay_rate;
  solver_out['smooth_eps'] = solver.smooth_eps;
  step_cache_out = {};
  for(var k in solver.step_cache) {
    step_cache_out[k] = solver.step_cache[k].toJSON();
  }
  solver_out['step_cache'] = step_cache_out;
  out['solver'] = solver_out;
  out['letterToIndex'] = letterToIndex;
  out['indexToLetter'] = indexToLetter;
  out['vocab'] = vocab;
  $("#tio").val(JSON.stringify(out));
}

var loadModel = (j) => {
  hidden_sizes = j.hidden_sizes;
  generator = j.generator;
  letter_size = j.letter_size;
  model = {};
  for(var k in j.model) {
    var matjson = j.model[k];
    model[k] = new Mat(1,1);
    model[k].fromJSON(matjson);
  }
  solver = new Solver(); // have to reinit the solver since model changed
  solver.decay_rate = j.solver.decay_rate;
  solver.smooth_eps = j.solver.smooth_eps;
  solver.step_cache = {};
  for(var k in j.solver.step_cache){
    var matjson = j.solver.step_cache[k];
    solver.step_cache[k] = new Mat(1,1);
    solver.step_cache[k].fromJSON(matjson);
  }
  letterToIndex = j['letterToIndex'];
  indexToLetter = j['indexToLetter'];
  vocab = j['vocab'];

  // reinit these
  ppl_list = [];
  tick_iter = 0;
}

let predict_worker = new WebWorker(work(require('./workers/predict_sentence_worker.js')));

var G = new Graph();

var costfun = (model, sent) => {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  G.reset();

  var n = sent.length;
  var log2ppl = 0.0;
  var cost = 0.0;
  var prev = {};

  for(var i=-1;i<n;i++) {
    // start and end tokens are zeros
    var ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
    var ix_target = i === n-1 ? 0 : letterToIndex[sent[i+1]]; // last step: end with END token

    lh = forwardIndex(G, model, ix_source, prev, generator, hidden_sizes);
    prev = lh;

    // set gradients into logprobabilities
    logprobs = lh.o; // interpret output as logprobs
    probs = softmax(logprobs); // compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);

    // write gradients into log probabilities
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1
  }
  var ppl = Math.pow(2, log2ppl / (n - 1));
  return {G, ppl, cost};
}

var ppl_list = [];

var cost_struct, solver_stats;

// worker for predicting sentences on separate thread


// time ticker for training and other tasks

var ticker = new Ticker(function() {
  // sample sentence from data
  let sent = _.sample(data_sents)

  // evaluate cost function on a sentence
  cost_struct = costfun(model, sent);
  
  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  
  // perform param update
  solver_stats = solver.step(model, learning_rate, regc, clipval);
  
  //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)
  ppl_list.push(cost_struct.ppl); // keep track of perplexity

  // evaluate now and then
});

ticker.every(50, function(){
  // draw samples
  predict_worker.send_work([predict_num_lines, [model, true, sample_softmax_temperature, letterToIndex, indexToLetter, logprobs, generator, hidden_sizes]], (result) => {
    $('#samples').html(result);
  });
});

ticker.every(10, function(){
  // draw argmax prediction

  predict_worker.send_work([1, [model, false, null, letterToIndex, indexToLetter, logprobs, generator, hidden_sizes]], (result) => {
    $('#argmax').html(`<div class="apred">${result}</div>`)
  });

  // keep track of perplexity
  $('#epoch').text('epoch: ' + (this.tick_iter/epoch_size).toFixed(2));
  $('#ppl').text('perplexity: ' + cost_struct.ppl.toFixed(2));
  $('#ticktime').text('forw/bwd time per example: ' + this.tick_time.toFixed(1) + 'ms');
});
  
ticker.every(100, function(){
  var median_ppl = median(ppl_list);
  ppl_list = [];
  pplGraph.add(this.tick_iter, median_ppl);
  pplGraph.drawSelf();
});

var gradCheck = () => {
  var model = initModel();
  var sent = '^test sentence$';
  var cost_struct = costfun(model, sent);
  cost_struct.G.backward();
  var eps = 0.000001;

  for(var k in model) {
    var m = model[k]; // mat ref
    for(var i=0,n=m.w.length;i<n;i++) {
      
      oldval = m.w[i];
      m.w[i] = oldval + eps;
      var c0 = costfun(model, sent);
      m.w[i] = oldval - eps;
      var c1 = costfun(model, sent);
      m.w[i] = oldval;

      var gnum = (c0.cost - c1.cost)/(2 * eps);
      var ganal = m.dw[i];
      var relerr = (gnum - ganal)/(Math.abs(gnum) + Math.abs(ganal));
      if(relerr > 1e-1) {
        console.log(k + ': numeric: ' + gnum + ', analytic: ' + ganal + ', err: ' + relerr);
      }
    }
  }
}

$(() => {

  // attach button handlers
  $('#learn').click(() => {
    pplGraph && pplGraph.reset();
    
    reinit().then(()=>{
      ticker.start();
    });
  });

  $('#stop').click(() => { 
    ticker.stop();
  });
  
  $("#resume").click(() => {
    ticker.start();
  });

  $("#savemodel").click(saveModel);

  $("#loadmodel").click(() => {
    var j = JSON.parse($("#tio").val());
    loadModel(j);
  });

  $("#loadpretrained").click(() => {
    $.getJSON("saved_states/lstm_100_model.json", (data) => {
      pplGraph = pplGraph || new Rvis.Graph();
      learning_rate = 0.0001;
      render_learning_slider();
      loadModel(data);
    });
  });

  $("#learn").click(); // simulate click on startup

  //$('#gradcheck').click(gradCheck);
  
  render_learning_slider();
  render_input_file_container();
  render_temperature_slider();
});

var set_input_file = (input_file) => {
  window.chosen_input_file = input_file;
  
  reinit().then(()=>{
    ticker.start();
  });
};

var learning_rate = 0.0001;

var render_learning_slider = () => {
  let callback = (value) => {
    learning_rate = Math.pow(10, value);
    render_learning_slider();
  };

  let transform_text = (value) => {
    return Math.pow(10, value).toFixed(5);
  };

  React.render(
    <Slider
      min={Math.log10(0.01) - 3.0}
      max={Math.log10(0.01) + 0.05}
      step={0.05}
      value={Math.log10(learning_rate)}
      slider_description="Learning rate: you want to anneal this over time if you're training for longer time."
      callback={callback}
      transform_text={transform_text}
    />
  , document.getElementById('learning_slider'));
}

var sample_softmax_temperature = 1;

var render_temperature_slider = () => {
  let callback = (value) => {
    sample_softmax_temperature = Math.pow(10, value);
    render_temperature_slider();  
  };

  let transform_text = (value)=> {
    return Math.pow(10, value).toFixed(2);
  }

  React.render(
    <Slider 
      value={Math.log10(sample_softmax_temperature)} 
      min={-1} 
      max={1.05} 
      step={0.05} 
      callback={callback} 
      transform_text={transform_text} 
      slider_description="Softmax sample temperature: lower setting will generate more likely predictions, but you'll see more of the same common words again and again. Higher setting will generate less frequent words but you might see more spelling errors."
    />,
    document.getElementById('temperature_slider')
  );
}

var render_input_file_container = () => {
  React.render(
    <ChooseInputTextComponent 
      input_files={window.input_files} 
      chosen_input_file={window.chosen_input_file} 
      callback={set_input_file}
      />, 
    document.getElementById('choose_input_file')
  );
}