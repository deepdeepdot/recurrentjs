import {forwardLSTM, initLSTM, forwardRNN, initRNN} from "./recurrent.js";
import Rvis from "./vis.js";
import {RandMat, Mat, softmax} from "./mat.js";
import {median, randi, maxi, samplei, gaussRandom} from "./math.js";
import Graph from "./graph.js";
import Solver from "./solver.js";
import React from "react";
import {classSet as cx} from "react-addons";
import _ from "lodash";
import $ from "jquery";

// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be
var max_chars_gen = 100; // max length of generated sentences

// various global var inits
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var data_sents = [];
var solver = new Solver(); // should be class because it needs memory for step caches
var pplGraph = new Rvis.Graph();

var generator, hidden_sizes, letter_size, regc, learning_rate, clipval, ch, lh, logprobs, probs, step_cache_out, step_cache;

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
  $("#prepro_status").text('found ' + vocab.length + ' distinct characters: ' + vocab.join(''));
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
  
  let network;
  
  if(generator === 'rnn') {
    network = initRNN(letter_size, hidden_sizes, output_size);
  } else {
    network = initLSTM(letter_size, hidden_sizes, output_size);
  }
  
  _.merge(model, network);
  return model;
}

var reinit_learning_rate_slider = () => {
  // init learning rate slider for controlling the decay
  // note that learning_rate is a global variable

  // #lr_slider values
  // min: Math.log10(0.01) - 3.0
  // max: Math.log10(0.01) + 0.05
  // step: 0.05
  // value: Math.log10(learning_rate)

  $("#lr_slider").on('input change', function(){
    learning_rate = Math.pow(10, this.value);
    $("#lr_text").text(learning_rate.toFixed(5));
  });

  $("#lr_text").text(learning_rate.toFixed(5));
}

window.chosen_input_file = window.input_files[randi(0, window.input_files.length)];

var reinit = (cb) => {
  // note: reinit writes global vars
  
  // eval options to set some globals
  stop_timer();

  eval($("#newnet").val());

  reinit_learning_rate_slider();

  solver = new Solver(); // reinit solver
  pplGraph = new Rvis.Graph("#pplgraph");

  ppl_list = [];
  tick_iter = 0;

  // process the input, filter out blanks

  $.get(`/data/${chosen_input_file}`, (text) => {
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
    cb && cb();
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

var forwardIndex = (G, model, ix, prev) => {
  var x = G.rowPluck(model['Wil'], ix);
  // forward prop the sequence learner
  if(generator === 'rnn') {
    var out_struct = forwardRNN(G, model, hidden_sizes, x, prev);
  } else {
    var out_struct = forwardLSTM(G, model, hidden_sizes, x, prev);
  }
  return out_struct;
}

var predictSentence = (model, samplei_bool=false, temperature=1.0) => {
  var G = new Graph(false);
  var s = '';
  var prev = {};
  while(true) {

    // RNN tick
    var ix = s.length === 0 ? 0 : letterToIndex[s[s.length-1]];
    var lh = forwardIndex(G, model, ix, prev);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
    if(temperature !== 1.0 && samplei_bool) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for(var q=0,nq=logprobs.w.length;q<nq;q++) {
        logprobs.w[q] /= temperature;
      }
    }

    probs = softmax(logprobs);
    if(samplei_bool) {
      var ix = samplei(probs.w);
    } else {
      var ix = maxi(probs.w);  
    }
    
    if(ix === 0) break; // END token predicted, break out
    if(s.length > max_chars_gen) { break; } // something is wrong

    var letter = indexToLetter[ix];
    s += letter;
  }
  return s;
}

var costfun = (model, sent) => {
  // takes a model and a sentence and
  // calculates the loss. Also returns the Graph
  // object which can be used to do backprop
  var n = sent.length;
  var G = new Graph();
  var log2ppl = 0.0;
  var cost = 0.0;
  var prev = {};

  for(var i=-1;i<n;i++) {
    // start and end tokens are zeros
    var ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
    var ix_target = i === n-1 ? 0 : letterToIndex[sent[i+1]]; // last step: end with END token

    lh = forwardIndex(G, model, ix_source, prev);
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
  return {'G':G, 'ppl':ppl, 'cost':cost};
}

var ppl_list = [];

var tick_iter = 0;

var stop_timer = () => {
  iid && clearTimeout(iid);
};

var tick = () => {
  stop_timer();
  iid = setTimeout(() => {
    // sample sentence from data
    var sentix = randi(0, data_sents.length);
    var sent = data_sents[sentix];

    var t0 = +new Date();  // log start timestamp

    // evaluate cost function on a sentence
    var cost_struct = costfun(model, sent);
    
    // use built up graph to compute backprop (set .dw fields in mats)
    cost_struct.G.backward();
    // perform param update
    var solver_stats = solver.step(model, learning_rate, regc, clipval);
    //$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)

    var t1 = +new Date();
    var tick_time = t1 - t0;

    ppl_list.push(cost_struct.ppl); // keep track of perplexity

    // evaluate now and then
    tick_iter += 1;

    if(tick_iter % 50 === 0) {
      // draw samples
      $('#samples').html('');
      for(var q=0;q<5;q++) {
        var pred = predictSentence(model, true, sample_softmax_temperature);
        var pred_div = '<div class="apred">'+pred+'</div>'
        $('#samples').append(pred_div);
      }
    }
    if(tick_iter % 10 === 0) {
      // draw argmax prediction
      $('#argmax').html('');
      var pred = predictSentence(model, false);
      var pred_div = '<div class="apred">'+pred+'</div>'
      $('#argmax').append(pred_div);

      // keep track of perplexity
      $('#epoch').text('epoch: ' + (tick_iter/epoch_size).toFixed(2));
      $('#ppl').text('perplexity: ' + cost_struct.ppl.toFixed(2));
      $('#ticktime').text('forw/bwd time per example: ' + tick_time.toFixed(1) + 'ms');

    }
    
    if(tick_iter % 100 === 0) {
      var median_ppl = median(ppl_list);
      ppl_list = [];
      pplGraph.add(tick_iter, median_ppl);
      pplGraph.drawSelf(document.getElementById("pplgraph"));
    }

    tick();
  }, 0);
}

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

var iid = null;

$(() => {

  // attach button handlers
  $('#learn').click(() => {
    pplGraph && pplGraph.reset();
    reinit(tick);
  });

  $('#stop').click(() => { 
    stop_timer();
  });
  
  $("#resume").click(() => {
    tick();
  });

  $("#savemodel").click(saveModel);

  $("#loadmodel").click(() => {
    var j = JSON.parse($("#tio").val());
    loadModel(j);
  });

  $("#loadpretrained").click(() => {
    $.getJSON("saved_states/lstm_100_model.json", (data) => {
      pplGraph = new Rvis.Graph();
      learning_rate = 0.0001;
      reinit_learning_rate_slider();
      loadModel(data);
    });
  });

  $("#learn").click(); // simulate click on startup

  $('#temperature_slider').on('input change', function(){
    sample_softmax_temperature = Math.pow(10, this.value);
    $("#temperature_text").text( sample_softmax_temperature.toFixed(2) );
  });

  //$('#gradcheck').click(gradCheck);

  render_input_file_container();
});

var ChooseInputTextComponent = React.createClass({
  load_input_file(input_file){
    window.chosen_input_file = input_file;
    reinit(tick);
  },

  render() {
    return (
      <ul className="ChooseInputTextComponent">
        {this.props.input_files.map((input_file, ind) => {
          let cls = cx({
            "sel": this.props.chosen_input_file==input_file
          });

          return (
            <li 
              key={`input-file-${ind}`} 
              onClick={this.load_input_file.bind(this, input_file)}
              className={cls}
            >
              {input_file}
            </li>
          );
        })}
      </ul>
    );
  }
});

var render_input_file_container = () => {
  React.render(
    <ChooseInputTextComponent input_files={window.input_files} chosen_input_file={window.chosen_input_file} />, 
    document.getElementById('choose_input_file')
  );
}