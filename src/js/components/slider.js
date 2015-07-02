import React from "react";
import {classSet as cx, PureRenderMixin} from "react-addons";

let Slider = React.createClass({
  mixins: [PureRenderMixin],

  changed(e){
    this.props.callback && this.props.callback(e.target.value);
  },

  render(){
    let text;
    
    if(this.props.transform_text){
      text = this.props.transform_text(this.props.value);
    } else {
      text = this.props.value;
    }

    return (
      <div className='Slider'>
        <div className="slider_header">{this.props.slider_description}</div>
        <input 
          type="range" 
          min={this.props.min} 
          max={this.props.max} 
          step={this.props.step} 
          value={this.props.value}
          onInput={this.changed}
          onChange={this.changed}
        />
        <div className="slider_value">{text}</div>
      </div>
    );
  }
});

export default Slider;