import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";
import PropTypes from 'prop-types';


import AddIcon from '@material-ui/icons/Add';
import CloseIcon from '@material-ui/icons/Close';
import IconButton from '@material-ui/core/IconButton';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import Switch from '@material-ui/core/Switch';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import TextField from '@material-ui/core/TextField';
import Tooltip from '@material-ui/core/Tooltip';
import Typography from '@material-ui/core/Typography';

import { highlight_regex, highlight, annotate, TokenToRegex, isToken } from './actions/annotate'

export const TOKEN_VIEW = 0;
export const REGEX_VIEW = 3;

class ConceptElement extends React.Component {
    constructor(props) {
        super(props);
        var token_text = this.props.token.string;
        if (isToken(this.props.token.string)) {
            token_text = isToken(this.props.token.string)
        }
        this.state = {
            type: this.props.view===TOKEN_VIEW ? this.props.keyType["TOKEN"] : this.props.keyType["REGEXP"], //default type. will be overrided if a type is provided.
            case_sensitive: false,
            ...this.props.token,
            token_text
        }

        this.availableKeyTypes = {}; // create a mapping {key_code: key_name}
        for (var i = ["TOKEN", "REGEXP"].length - 1; i >= 0; i--) {
            let keytype_name = ["TOKEN", "REGEXP"][i];
            this.availableKeyTypes[this.props.keyType[keytype_name]] = keytype_name;
        }

    }

    handleKeyPress(event){ 
        if (event.key === 'Enter') {
            this.submit();
        }
    }

    resetHighlights() {
        this.props.highlight([]);
    }

    handleInput(event) {
        const raw_str = event.target.value;
        var regex = raw_str;
        var newState = {};
        if (this.state.type === this.props.keyType["TOKEN"]) { // it's a token
            if (this.props.view === TOKEN_VIEW) { // token view
                newState = {
                    string: TokenToRegex(regex),
                    token_text: event.target.value
                }                
            } else { //regex view
                newState = {
                    type: this.props.keyType["REGEXP"],
                    string: event.target.value,
                    token_text: event.target.value
                }
            }
        } else { // it's a regexp. view doesn't matter.
            newState = {
                string: event.target.value,
                token_text: event.target.value
            }
        }
        // annotate
        this.refreshHighlights(newState.string);
        this.setState(newState);
    }

    changeKeytype(new_type) {
        var newState = {};
        if (new_type===0) { // regexp change to token
            newState = {
                type: this.props.keyType[new_type],
                token_text: this.state.string,
                string: TokenToRegex(this.state.string)
            };
        } else { // token change to regexp
            newState = {
                type: this.props.keyType[new_type],
                string: this.state.token_text,
                token_text: this.state.token_text
            };
        }
        this.refreshHighlights(newState.string);
        this.setState(newState);
    }

    refreshHighlights(regex_string=this.state.string, case_sensitive=this.state.case_sensitive) {
        if (!this.isEmpty()) {
            this.props.highlight_regex(regex_string, this.props.text, case_sensitive, this.props.idx);
        } else {
            this.resetHighlights();
        }
    }

    changeCaseSensitivity() {
        const bool = this.state.case_sensitive;
        this.refreshHighlights(this.state.string, !bool);
        this.setState({case_sensitive: !bool});
    }

    submit() {        
        if (this.isValid()) {
            if (this.state!==this.props.token) {
                this.props.changeToken(this.state);
                // annotate the text with the new condition, if applicable
                const hlights = this.props.highlights.map(hlight => {
                    hlight.id = hlight.start_offset;
                    hlight.label = this.props.concept;
                    hlight.origin = this.state.string;
                    return hlight;
                });
                this.props.annotate(this.props.addAnnotations(hlights));
            }
        }
        this.resetHighlights();   
    }

    isEmpty(state=null){
        if (state===null) {
            state = this.state;
        }
        if (state.string==="") {
            return true;
        }
        if (state.token_text==="") {
            return true;
        }
        return false
    }

    onBlur() { // Event triggered when textfield loses focus (click outside, for example)
        if (!this.props.new_entry) {
            this.submit();
        }
    }

    isValid() {
        if (this.isEmpty()) {
            return false
        }
        if (this.props.error) {
            return false;
        }
        return true;
    }

    render() {
        const keytype = this.state.type;
        var SelectField = null;
        if (this.props.view===TOKEN_VIEW) {
            SelectField = <Select
                            value={this.availableKeyTypes[keytype]}
                            disabled={false}
                            onChange={(event) => this.changeKeytype(event.target.value)}>
                                {Object.values(this.availableKeyTypes).map((ktype) => 
                                <MenuItem key={ktype} value={ktype}>
                                    <Typography variant="body2">{ktype.toLowerCase()}</Typography>
                                </MenuItem>)}
                        </Select>
        }
        return (
            <TableRow key={this.props.idx} onBlur={this.onBlur.bind(this)}>
                <TableCell>
                    <TextField
                        //fullWidth={true}
                        className={this.props.classes.input}
                        value={this.props.view===TOKEN_VIEW ? this.state.token_text : this.state.string }
                        onKeyPress = {this.handleKeyPress.bind(this)}
                        onChange = {this.handleInput.bind(this)}
                        error={(this.props.error !== undefined)}
                        helperText={this.props.error}
                        autoFocus={this.props.new_entry ? true : false}
                    >
                    </TextField>
                </TableCell>
                {this.props.view===TOKEN_VIEW ? <TableCell padding="none">
                    {SelectField}
                </TableCell>: null}
                <TableCell padding="none">
                    <Tooltip title="Toggle case-sensitivity" enterDelay={500}>
                        <Switch checked={this.state.case_sensitive} onChange={this.changeCaseSensitivity.bind(this)}/>
                    </Tooltip>
                </TableCell>
                <TableCell padding="none">
                    <IconButton  
                    //color="primary" 
                    size="small"
                    onClick={this.props.delete ? this.props.delete : this.submit.bind(this)}
                    aria-label={this.props.delete ? "delete element" : "add element"} >
                        {this.props.delete ? <CloseIcon/> : <AddIcon/>}
                    </IconButton>
                </TableCell>
            </TableRow>
        )
    }
}

ConceptElement.propTypes = {
    idx: PropTypes.number.isRequired,
    token: PropTypes.object.isRequired,
    changeToken: PropTypes.func.isRequired,
    view: PropTypes.number.isRequired,
}

function mapStateToProps(state, ownProps?) {
    const error = state.highlights.error[ownProps.idx];
    return { 
        text: state.text.data,
        error: error,
        highlights: state.highlights.data,
        keyType: state.gll.keyType,
    };
}

function mapDispatchToProps(dispatch) {
    return { 
        annotate: bindActionCreators(annotate, dispatch), 
        highlight: bindActionCreators(highlight, dispatch),
        highlight_regex: bindActionCreators(highlight_regex, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(ConceptElement);