import _ from "lodash";
import $ from "jquery";
import work from "webworkify";

import {costfun, forwardIndex, forwardLSTM, initLSTM, forwardRNN, initRNN} from "./recurrent";
import Rvis from "./vis";
import {RandMat, Mat} from "./mat";
import softmax from "./softmax"
import {median, randi, maxi, samplei, gaussRandom} from "./math";
import Graph from "./graph";
import Solver from "./solver";
import Ticker from "./ticker";
import WebWorker from "./webworker";

import React from "react";
import {classSet as cx} from "react-addons";
import ChooseInputTextComponent from "./components/choose_input_text_component";
import Slider from "./components/slider";

// prediction params
let sample_softmax_temperature = 1.0; // how peaky model predictions should be

// letious global let inits
let epoch_size, input_size, output_size;
let letterToIndex = {};
let indexToLetter = {};
let vocab = [];
let data_sents = [];

let pplGraph;

let step_cache_out, step_cache, input_text;

let predict_num_lines = 10; // number of lines for the prediction to show
let model = {};

let generator = 'lstm';       // can be 'rnn' or 'lstm'
let hidden_sizes = [20, 20];   // list of sizes of hidden layers
let letter_size = 5;          // size of letter embeddings

// optimization
let regc = 0.000001;          // L2 regularization strength
let learning_rate = 0.01;     // learning rate
let clipval = 5.0;            // clip gradients at this value


let sampler  = new WebWorker(work(require('./workers/predict_sentence_worker.js')));
let trainer  = new WebWorker(work(require('./workers/train_model_worker.js')));

let epoch, perplexity, ticktime, samples, argmax;

let initVocab = (sents, count_threshold) => {
  // go over all characters and keep track of all unique ones seen
  let txt = sents.join(''); // concat all

  // count up all characters
  let d = {};
  for(let i=0,n=txt.length;i<n;i++) {
    let txti = txt[i];
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
  let q = 1; 
  for(let ch in d) {
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
  vocab.sort();
  return vocab;
}

let App = React.createClass({

  componentDidMount() {
    this.startLearning();
  },

  registerSamplers() {
    this.stopSamplers();

    this.sample_loop = setInterval(()=>{
      trainer.send(["sample_model"]).then((result)=>{
        model = result.model;
        
        sampler.send([1, [model, false, null, letterToIndex, indexToLetter, generator, hidden_sizes]])
          .then((result) => {
            argmax = result;
            this.forceUpdate();
          });
        
        sampler.send([predict_num_lines, [model, true, sample_softmax_temperature, letterToIndex, indexToLetter, generator, hidden_sizes]])
          .then((result) => {
            samples = result;
            this.forceUpdate();
          });
      });
    }, 1000);

    this.draw_loop = setInterval(()=>{
      trainer.send(["sample_ppl"]).then(({tick_iter, ppl_list})=>{
        if(ppl_list.length < 10){ return; }
        let median_ppl = median(ppl_list);
        ppl_list = [];
        pplGraph.add(tick_iter, median_ppl);
        pplGraph.drawSelf();
      });
    }, 2500);
    
    this.sample_data_loop = setInterval(()=>{
      trainer.send(["sample"]).then((result)=>{
        epoch = (result.tick_iter/epoch_size).toFixed(2);
        perplexity = result.ppl.toFixed(2);
        ticktime = result.tick_time.toFixed(1);
        this.forceUpdate();
      });
    }, 500);
  },

  stopSamplers() {
    this.sample_data_loop && clearInterval(this.sample_data_loop);
    this.sample_loop && clearInterval(this.sample_loop);
    this.draw_loop && clearInterval(this.draw_loop);
  },

  startLearning() {
    this.stopSamplers();
    pplGraph && pplGraph.reset();
    
    this.reinit().then(()=>{
      this.resumeLearning();
    });
  },

  stopLearning() {
    this.stopSamplers();
    trainer.send(["stop"]);
  },

  resumeLearning() {
    trainer.send(["start"]);
    this.registerSamplers();
  },

  initModel() {
    let model = {
      Wil: new RandMat(input_size, letter_size)
    };
    
    let fn = (generator==='rnn') ? initRNN : initLSTM;
    let network = fn(letter_size, hidden_sizes, output_size)
    
    _.merge(model, network);

    trainer.send(["init", {
      model,
      generator,
      hidden_sizes,
      data_sents,
      learning_rate,
      regc,
      clipval,
      letterToIndex
    }]);

    return model;
  },

  saveModel() {
    let out = {};
    out['hidden_sizes'] = hidden_sizes;
    out['generator'] = generator;
    out['letter_size'] = letter_size;
    let model_out = {};
    for(let k in model) {
      model_out[k] = model[k].toJSON();
    }
    out['model'] = model_out;
    let solver_out = {};
    solver_out['decay_rate'] = solver.decay_rate;
    solver_out['smooth_eps'] = solver.smooth_eps;
    step_cache_out = {};
    for(let k in solver.step_cache) {
      step_cache_out[k] = solver.step_cache[k].toJSON();
    }
    solver_out['step_cache'] = step_cache_out;
    out['solver'] = solver_out;
    out['letterToIndex'] = letterToIndex;
    out['indexToLetter'] = indexToLetter;
    out['vocab'] = vocab;
    $("#tio").val(JSON.stringify(out));
  },

  loadModel() {
    let j = JSON.parse($("#tio").val());
    hidden_sizes = j.hidden_sizes;
    generator = j.generator;
    letter_size = j.letter_size;
    model = {};
    for(let k in j.model) {
      let matjson = j.model[k];
      model[k] = new Mat(1,1);
      model[k].fromJSON(matjson);
    }
    let solver = new Solver(); // have to reinit the solver since model changed
    solver.decay_rate = j.solver.decay_rate;
    solver.smooth_eps = j.solver.smooth_eps;
    solver.step_cache = {};
    for(let k in j.solver.step_cache){
      let matjson = j.solver.step_cache[k];
      solver.step_cache[k] = new Mat(1,1);
      solver.step_cache[k].fromJSON(matjson);
    }
    letterToIndex = j['letterToIndex'];
    indexToLetter = j['indexToLetter'];
    vocab = j['vocab'];

    trainer.send(["reset"]);
  },

  loadPretrained() {
    $.getJSON("saved_states/lstm_100_model.json", (data) => {
      pplGraph = pplGraph || new Rvis.Graph();
      learning_rate = 0.0001;
      this.forceUpdate();
      loadModel(data);
    });
  },

  reinit() {
    // note: reinit writes global lets
    // eval options to set some globals

    epoch = null;
    ticktime = null;
    perplexity = null;

    pplGraph = pplGraph || new Rvis.Graph("#pplgraph");
    pplGraph.reset();

    trainer.send(["reset"]);

    // process the input, filter out blanks

    return Promise.resolve($.get(`/data/${chosen_input_file}`)).then((text) => {
      input_text = text;

      let data_sents_raw = text.split('\n');

      data_sents = [];

      for(let i=0;i<data_sents_raw.length;i++) {
        let sent = data_sents_raw[i].trim();
        if(sent.length > 0) {
          data_sents.push(sent);
        }
      }

      initVocab(data_sents, 1); // takes count threshold for characters
      this.forceUpdate();

      model = this.initModel();
    });
  },

  set_input_file(input_file) {
    window.chosen_input_file = input_file;
    
    this.reinit().then(()=>{
      this.resumeLearning();
    });
  },

  render() {
    let callback = (value) => {
      sample_softmax_temperature = Math.pow(10, value);
      this.forceUpdate();
    };

    let temperature_slider = (
      <Slider 
        value={Math.log10(sample_softmax_temperature)} 
        min={-1} 
        max={1.05} 
        step={0.05} 
        callback={callback} 
        transform_text={(value)=> Math.pow(10, value).toFixed(2)} 
        slider_description="Softmax sample temperature: lower setting will generate more likely predictions, but you'll see more of the same common words again and again. Higher setting will generate less frequent words but you might see more spelling errors."
      />
    );

    let learning_callback = (value) => {
      learning_rate = Math.pow(10, value);
      trainer.send(["update_learning_rate", {learning_rate}]);
      this.forceUpdate();
    };

    let learning_slider = (
      <Slider
        min={Math.log10(0.01) - 3.0}
        max={Math.log10(0.01) + 0.05}
        step={0.05}
        value={Math.log10(learning_rate)}
        slider_description="Learning rate: you want to anneal this over time if you're training for longer time."
        callback={learning_callback}
        transform_text={(value) => Math.pow(10, value).toFixed(5)}
      />
    );

    let choose_input_file_component = (
      <ChooseInputTextComponent 
        input_files={window.input_files} 
        chosen_input_file={window.chosen_input_file} 
        callback={this.set_input_file}
      />
    );

    let prepos_text = `found ${vocab.length} distinct characters: ${vocab.join('')}`
    let epoch_text = `epoch: ${epoch}`;
    let perplexity_text = `perplexity: ${perplexity}`;
    let ticktime_text = `forw/bwd time per example: ${ticktime} ms`;

    return (
      <div className="AppComponent">
        <div>
          <div className="hh">Input files:</div>
          <div id="choose_input_file">{choose_input_file_component}</div>
          <div className="hh">Input sentences:</div>
          <div id="prepro_status">{prepos_text}</div>
          <div id="input_text">{input_text}</div>
        </div>

        <div className="hh">Controls/Options:</div>
        
        <button id="learn" className="abutton" onClick={this.startLearning}>learn/restart</button>
        <button id="resume" className="abutton" onClick={this.resumeLearning}>resume</button>
        <button id="stop" className="abutton" onClick={this.stopLearning}>pause</button>

        <p>
          protip: if your perplexity is exploding with Infinity try lowering the initial learning rate
        </p>
        
        <div id="status">
          <div>
            <div className="hh">Training stats:</div>
            <div className="aslider" id="learning_slider">{learning_slider}</div>

            <div className="clearfix">
              <div className="lt">
                <div id="ticktime">{ticktime && ticktime_text}</div>
                <div id="epoch">{epoch && epoch_text}</div>
                <div id="ppl">{perplexity && perplexity_text}</div>
              </div>
              <div id="pplgraph"></div>
            </div>
          </div>

          <div className="hh">Model samples:</div>
          <div id="controls">
            <div className="aslider" id="temperature_slider">{temperature_slider}</div>
          </div>
          <div id="samples">{samples}</div>
          <div className="hh">Greedy argmax prediction:</div>
          <div id="argmax"><div className="apred">{argmax}</div></div>
        </div>

        <div id="io">
          <div className="hh">I/O save/load model JSON</div>

          <button id="savemodel" className="abutton" onClick={this.saveModel}>save model</button>
          <button id="loadmodel" className="abutton" onClick={this.loadModel}>load model</button>
          <div>
            You can save or load models with JSON using the textarea below.
          </div>
          <textarea id="tio"></textarea>

          <div className="hh">Pretrained model:</div>
          <p>
            You can also choose to load an example pretrained model with the button below to see what the predictions look like in later stages. The pretrained model is an LSTM with one layer of 100 units, trained for ~10 hours. After clicking button below you should see the perplexity plummet to about 3.0, and see the predictions become better.
          </p>
          <button id="loadpretrained" className="abutton" onClick={this.loadPretrained}>load pretrained</button>
        </div>
      </div>
    );
  }
})

window.chosen_input_file = _.sample(window.input_files);

React.render(
  <App/>, 
  document.getElementById('application')
);

// let gradCheck = () => {
//   let model = initModel();
//   let sent = '^test sentence$';
//   let cost_struct = costfun(model, sent);
//   cost_struct.G.backward();
//   let eps = 0.000001;

//   for(let k in model) {
//     let m = model[k]; // mat ref
//     for(let i=0,n=m.w.length;i<n;i++) {
      
//       oldval = m.w[i];
//       m.w[i] = oldval + eps;
//       let c0 = costfun(model, sent);
//       m.w[i] = oldval - eps;
//       let c1 = costfun(model, sent);
//       m.w[i] = oldval;

//       let gnum = (c0.cost - c1.cost)/(2 * eps);
//       let ganal = m.dw[i];
//       let relerr = (gnum - ganal)/(Math.abs(gnum) + Math.abs(ganal));
//       if(relerr > 1e-1) {
//         console.log(k + ': numeric: ' + gnum + ', analytic: ' + ganal + ', err: ' + relerr);
//       }
//     }
//   }
// }

//$('#gradcheck').click(gradCheck);