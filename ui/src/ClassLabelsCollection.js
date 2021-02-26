import React from 'react';
import Button from '@material-ui/core/Button';
import ButtonGroup from '@material-ui/core/ButtonGroup';
import Grid from '@material-ui/core/Grid';
import {connect} from "react-redux";

class ClassLabelsCollection extends React.Component {
    constructor(props) {
      super(props)
      this.keyHandling = this.keyHandling.bind(this);
      this.assignLabel = this.assignLabel.bind(this);
    }

    defaultSelections

    keyHandling(e) {
      const key = parseInt(e.key);
      if (key in this.props.hotKeys)  {
        this.assignLabel(key);
      }
    }

    assignLabel(key) {
      this.props.onClick(key);
    }

    componentDidMount() {     
      window.addEventListener("keyup", this.keyHandling);   
    }        

    componentWillUnmount() {      
      window.removeEventListener("keyup", this.keyHandling);   
    }

    render(){
        const classes = this.props.classes;
        //console.log(this.props.labelClasses)
        return (
            <Grid>
             { this.props.labelClasses.map( (labelClass) => 
                   <Button 
                        className={classes.button + String(labelClass.key)} 
                        key = {labelClass.key}
                        onClick={() => this.assignLabel(labelClass.key)} 
                        style={
                          {backgroundColor:labelClass.backgroundColor,
                          color:labelClass.textColor,
                          fontWeight:'bold',
                          margin:'5px',
                        }} 
                        size={"small"}
                    >
                        ({labelClass.key})
                        {" "+ labelClass.name}
                    </Button>
                  )
              }
            </Grid>)
    }   
}

function mapStateToProps(state, ownProps?) {
    return { 
      labelClasses: state.labelClasses.data,
      hotKeys: state.labelClasses.data.map(lClass => lClass.key),
      annotations: state.annotations,
      label: state.label
    };
}
function mapDispatchToProps(state){
  return {};
}

export default connect(mapStateToProps, mapDispatchToProps)(ClassLabelsCollection);