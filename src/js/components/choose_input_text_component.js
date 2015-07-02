import React from "react";
import {classSet as cx} from "react-addons";

let ChooseInputTextComponent = React.createClass({

  load_input_file(input_file){
    this.props.callback && this.props.callback(input_file);
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

export default ChooseInputTextComponent;