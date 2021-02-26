import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

import fetchClasses, {addLabelClass} from './actions/labelClasses';

import AddIcon from '@material-ui/icons/Add';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';

const colorTheme = [
    {
        backgroundColor: 'rgb(255,236,153)',
        textColor: "rgb(239,140,2)",
    },
    {
        backgroundColor: "rgb(183,217,179)",
        textColor: "rgb(43,138,62)",
    },
    {
        backgroundColor: "rgb(201,231,239)",
        textColor: "rgb(12,152,173)",
    },
    {
        backgroundColor: "rgb(227,217,237)",
        textColor: "rgb(83,75,154)",
    },
    {
        backgroundColor: "rgb(248,200,201)",
        textColor: "rgb(225,49,49)",
    },
]

class LabelCreation extends React.Component {
    constructor(props) {
        super(props);
        if (this.props.labelClassesPending) {
            this.props.fetchClasses();
        }
        var new_label_key = Object.keys(props.labelClasses).length;

        this.state = {
          newLabel: "",
          fetched: false,
          new_label_key: new_label_key
        };
        this.handleInput = this.handleInput.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);
        this.handleAdd = this.handleAdd.bind(this);

    }

    handleInput(event) {
        this.setState({
            newLabel: event.target.value
        });
    }


    handleAdd() {
        const newLabel = this.state.newLabel.trim()
        if (newLabel !== "") {
            let lkey = Object.keys(this.props.labelClasses).length % colorTheme.length
            this.props.addLabel({key: lkey,
                name:newLabel,
                backgroundColor:colorTheme[lkey].backgroundColor,
                textColor:colorTheme[lkey].textColor
            });
            this.setState({
                newLabel: "", 
                new_label_key: Object.keys(this.props.labelClasses).length + 1
            });
        }
    }

    handleKeyPress(event){ 
        if (event.key === 'Enter') {
            this.handleAdd();
        }
    }                  

    render() {
        const labels = Object.values(this.props.labelClasses);
        const classes = this.props.classes;

        return(
            <div>
                <Typography>What would you like to name this project?</Typography>
                <div  className={classes.contents}>
                    <TextField  
                        required
                        className={classes.input}
                        placeholder="project name"
                        inputProps={{ 'aria-label': 'name your project' }}
                    />
                </div>
                <Typography>Name your label classes.</Typography>
                <div  className={classes.contents}>
                {   labels.map( (item) => {
                        console.log(item);
                        var lname = item.name;
                        var value = item.key;
                        return(
                            <div key={value}>
                            <Button 
                                className={classes.button} 
                                key = {value}
                                variant="outlined"
                            >
                                ({value})
                                {lname}
                            </Button>
                            </div>
                        )
                    })
                }
                    <TextField
                       className={classes.input}
                       placeholder="add label class"
                       inputProps={{ 'aria-label': 'add label class' }}
                       onKeyPress = {this.handleKeyPress}
                       onChange = {this.handleInput}
                       value={this.state.newLabel}
                       InputProps={{endAdornment:
                        <Button className={classes.button} size="small" color="primary" aria-label="add" 
                            onClick={this.handleAdd} disabled={(this.state.newLabel==="")} >
                            <AddIcon />
                        </Button>}}
                    />
                </div>
            </div>
        )
    }
}

function mapStateToProps(state, ownProps?) {
    return { 
        labelClasses: state.labelClasses.data,
        hotKeys: state.labelClasses.data,
        labelClassesPending: state.labelClasses.pending,
    };
}

function mapDispatchToProps(dispatch){
    // TODO action ADDLABEL
    return {
        addLabel: bindActionCreators(addLabelClass, dispatch),
        fetchClasses: bindActionCreators(fetchClasses, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(LabelCreation);