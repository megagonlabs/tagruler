import React from 'react';
import Button from '@material-ui/core/Button';
import {connect} from "react-redux";
import {bindActionCreators} from 'redux';
import PropTypes from 'prop-types';

import Checkbox from '@material-ui/core/Checkbox';
import InfoIcon from '@material-ui/icons/Info';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Tooltip from '@material-ui/core/Tooltip';
import Typography from '@material-ui/core/Typography';
import WarningIcon from '@material-ui/icons/Warning';

import { set_selected_LF } from './actions/labelAndSuggestLF'

class LabelingFunctionsSuggested extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            all_selected: false
        };

    }


    componentDidUpdate(prevProps) {
        if (this.state.all_selected) {
            for (var i = Object.values(this.props.labelingFunctions).length - 1; i >= 0; i--) {
                let lf = Object.values(this.props.labelingFunctions)[i];
                if (lf.selected !== true) {
                    this.setState({all_selected: false})
                }
            }
        }
    }

    label(lf) {
        return (
            this.props.labelClasses
                .filter(c => c.key === lf.Label)[0].name
        );    
    }

    conditionToString(condition) {
        let string = condition["string"];
        if (condition["case_sensitive"]) {
            string = "<b>"+string+"</b>";
        }
        if (condition.type === this.props.keyType["CONTEXT_SIMILAR"]){
            string = "\""+string +"\""
        }

        let condition_type_text = <span>{condition.TYPE_}</span>
        if ("explanation" in condition) { // Add mouseover explanation, if available
            condition_type_text = <Tooltip title={condition.explanation} aria-label={condition.explanation}>{condition_type_text}</Tooltip>
        } 

        if (condition.positive) {
            return <>{string} ({condition_type_text})</>;
        } else {
            return <>NOT({string}) ({condition_type_text})</>;
        }
    }

    conditions(lf) {
        const conditions = lf.Conditions.map(cond => this.conditionToString(cond));
        return conditions
            .reduce((prev, curr) => [prev, " " + lf.CONNECTIVE_ + " ", curr])
    }

    LFtoStrings(key, lf) {
        const stringsDict = {
            id: key,
            conditions: this.conditions(lf),
            context: lf.CONTEXT_,
            label: this.label(lf),
            order: lf.Direction.toString(),
            weight: lf.Weight,
            target: lf.Target
        };
        return stringsDict;
    }

    selectAllLF(bool_selected) {
        // (de)select all LFs, depending on value of bool_selected
        const LF_names = Object.keys(this.props.labelingFunctions);

        let newLFs = {};
        for (var i = LF_names.length - 1; i >= 0; i--) {
            let LF_key = LF_names[i];
            newLFs[LF_key] = this.props.labelingFunctions[LF_key];
            newLFs[LF_key]['selected'] = bool_selected;
        }
    
        this.setState({all_selected: bool_selected});
        this.props.set_selected_LF(newLFs);
    }

    handleChange(name, event) {
        let updatedLF = this.props.labelingFunctions[name];
        updatedLF['selected'] = !(updatedLF['selected']);
        const newLFs = {
            ...this.props.labelingFunctions,
            [name]: updatedLF 
        };
        this.props.set_selected_LF(newLFs);
    }



    render() {
        const classes = this.props.classes;

        var show_context = false;
        const LFList = Object.keys(this.props.labelingFunctions).map((lf_key) => {
            var lf_dict = this.LFtoStrings(lf_key, this.props.labelingFunctions[lf_key])
            if (lf_dict.context) {
                show_context = true;
            }
            return lf_dict;
        });
        var LF_content = <Table size="small" aria-label="suggested labeling functions table">
              <TableHead>
                <TableRow>
                  <TableCell>
                    <Checkbox
                        onChange={(event) => this.selectAllLF(!this.state.all_selected)}
                        checked={this.state.all_selected}
                    /> 
                    { this.state.all_selected ? "Deselect All" : "Select All"}
                  </TableCell>
                  <TableCell align="right">Token</TableCell>
                  <TableCell align="right">Conditions</TableCell>
                  { show_context ? <TableCell align="right">Context</TableCell> : null}                  
                  <TableCell align="center">Label</TableCell>
                  {/*<TableCell align="right">Reliability</TableCell>*/}
                </TableRow>
              </TableHead>
              <TableBody>
                {LFList.map(row => (
                  <TableRow key={Object.values(row).join('')}>
                    <TableCell component="th" scope="row">
                      <Checkbox 
                        key={this.props.labelingFunctions[row.id].selected}
                        onChange={(event) => this.handleChange(row.id, event)} 
                        checked={this.props.labelingFunctions[row.id].selected===true}/>
                    </TableCell>
                    <TableCell align="right">{row.target}</TableCell>
                    <TableCell align="right">{row.conditions}</TableCell>
                    {/*show_context ? <TableCell align="right">{row.context}</TableCell> : null*/}
                    {/*<TableCell align="right">{row.order}</TableCell>*/}
                    {/*<TableCell align="right">{row.label}</TableCell>*/}
                    <TableCell align="center">
                        <Button 
                            className={classes.button + this.props.labelClasses.filter(l => l.name === row.label)[0].name} 
                            key = {this.props.labelClasses.filter(l => l.name === row.label)[0].key}
                            style={
                            {backgroundColor:this.props.labelClasses.filter(l => l.name === row.label)[0].backgroundColor,
                            color:this.props.labelClasses.filter(l => l.name === row.label)[0].textColor,
                            fontWeight:'bold',
                            margin:'5px',
                            }} 
                            size={"small"}
                        >
                            {this.props.labelClasses.filter(l => l.name === row.label)[0].name}
                        </Button>
                    </TableCell>
                    {/*<TableCell align="right">{(row.weight).toFixed(2)}</TableCell>*/}
                  </TableRow>
                ))}
              </TableBody>
            </Table>

        return(
          <Paper className={classes.paper}>
            <Typography className={classes.title} variant="h6" id="tableTitle">
                Suggested Labeling Functions
            </Typography>
            { this.props.no_label ? <Typography variant="body1" color="error"><WarningIcon/>{"You must assign a label in order to generate labeling functions!"}</Typography> : "" }
            { (this.props.no_annotations && !(this.props.no_label)) ?  <Typography variant="body1"><InfoIcon/>{"TIP: to improve function suggestions, annotate the parts of the text that guided your decision."}</Typography> : "" }
            {LF_content}
          </Paper>
        );
    }
}

LabelingFunctionsSuggested.propTypes = {
    all_selected: PropTypes.bool
};

function mapStateToProps(state, ownProps?) {

    return { 
        labelingFunctions: state.suggestedLF,
        labelClasses:state.labelClasses.data, 
        no_annotations: (state.annotations.length < 1),
        no_label: (state.label === null),
        keyType: state.gll.keyType
    };
}

function mapDispatchToProps(dispatch) {
    return {
        set_selected_LF: bindActionCreators(set_selected_LF, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(LabelingFunctionsSuggested);