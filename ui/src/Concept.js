import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

import Avatar from '@material-ui/core/Avatar';
import Badge from '@material-ui/core/Badge';
import Card from '@material-ui/core/Card';
import CardHeader from '@material-ui/core/CardHeader';
import Collapse from '@material-ui/core/Collapse';
import DeleteForeverIcon from '@material-ui/icons/DeleteForever';
import IconButton from '@material-ui/core/IconButton';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';

import ConceptElement from './ConceptElement'
import { conceptEditors, select_concept } from './actions/concepts'
import { annotate, highlight } from './actions/annotate'

import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpandLessIcon from '@material-ui/icons/ExpandLess';

class Concept extends React.Component {
    constructor(props) {
        super(props);
        this.toggleOpen = this.toggleOpen.bind(this);
        this.handleRemove = this.handleRemove.bind(this);
        this.handleClick = this.handleClick.bind(this);
    } 

    changeToken(token_idx, new_token) {
        if (this.props.tokens[token_idx] !== new_token) {
            var current_tokens = this.props.tokens;
            current_tokens[token_idx] = new_token;
            this.updateConcept(current_tokens);        
        }
    }

    handleRemove(old_token_idx) {
        const old_token = this.props.tokens[old_token_idx];
        var current_tokens = this.props.tokens;
        current_tokens = current_tokens.filter((elt, idx) => idx !== old_token_idx);
        this.updateConcept(current_tokens);

        //Remove annotations that came from this element
        var new_annotations = this.props.annotations.filter(
            annotation => {
                if (annotation.origin == old_token.string) {
                    return false;
                } return true;
            })
        this.props.annotate(new_annotations);
    }

    updateConcept(tokens) {
        this.props.conceptEditors.updateConcept(
            this.props.concept,
            tokens
        );
    }
    
    toggleOpen(event){
        if (this.props.isOpen) {
            this.props.close();
        } else {
            this.props.open();
        }
    }

    handleClick(event){
        if (this.props.selectedConcept===this.props.concept) {
            //this.props.select_concept(null);
        } else {
            this.props.select_concept(this.props.concept);
        }
    }

    deleteSelf() {
        this.props.select_concept(null);
        this.props.conceptEditors.deleteConcept(this.props.concept);
    }

    render() {
        const classes = this.props.classes;
        const concept = this.props.concept;
        const color = this.props.color;

        const isSelected = (this.props.selectedConcept===this.props.concept);
        let style = {padding: 5, border: "2px solid " + color}
        if (isSelected) {
            style["backgroundColor"] = color;
            style["color"] = "white";
        } else {
            style["backgroundColor"] = "white";
            style["color"] = "black";
        }

        const new_token_idx = this.props.tokens.length;

        return(
        <div>
            <Card>
                <CardHeader
                    onClick = {this.handleClick}
                    style = { style }
                    action={
                        <><IconButton 
                            aria-label="delete concept" 
                            onClick={this.deleteSelf.bind(this)}>
                            <DeleteForeverIcon/>
                        </IconButton>
                        <IconButton
                            onClick={this.toggleOpen}
                            aria-expanded={this.props.isOpen}
                            aria-label="show more">
                            {this.props.isOpen ? <ExpandLessIcon/> : <ExpandMoreIcon/>}
                        </IconButton> </> 
                    }
                    title={<Typography>{concept}</Typography>}
                    avatar={<Avatar 
                        aria-label={concept} 
                        className={classes.avatar} size="small"  >
                            {new_token_idx}
                    </Avatar>}
                />
                <Collapse in={this.props.isOpen} timeout="auto" unmountOnExit>
                    <Table size="small" stickyHeader>
                        <TableHead>
                        <TableRow>
                            <TableCell>value</TableCell>
                            {this.props.view ? null : <TableCell>type</TableCell>}
                            <TableCell>case</TableCell>
                            <TableCell></TableCell>
                        </TableRow>
                        </TableHead>
                        <TableBody>
                    {this.props.tokens.map((token, idx) => {
                        let key = token.string + String(token.type) + String(token.case_sensitive) + String(idx);
                        return (<ConceptElement 
                                    token={token}
                                    delete={() => this.handleRemove(idx)}
                                    changeToken={(new_token) => this.changeToken(idx, new_token)} 
                                    key={key}
                                    idx={idx}
                                    concept={concept}
                                    addAnnotations={this.props.addAnnotations}
                                    classes={classes}
                                    view={this.props.view}
                                />)
                    })}
                    <ConceptElement 
                        token={{string: "", token_text: ""}}
                        changeToken={(new_token) => this.changeToken(new_token_idx, new_token)} 
                        key={new_token_idx}
                        idx={new_token_idx}
                        classes={classes}
                        concept={concept}
                        view={this.props.view}
                        addAnnotations={this.props.addAnnotations}
                        new_entry={true}
                    />
                    </TableBody></Table>
                </Collapse>
            </Card>
        </div>
        );
    }
}

function mapStateToProps(state, ownProps?) {
    let concept = state.concepts.data[ownProps.concept];
    return { 
        ...concept,
        selectedConcept: state.selectedConcept,
        text: state.text.data,
        error: state.highlights.error,
        highlights: state.highlights.data,
        annotations: state.annotations
    };
}

function mapDispatchToProps(dispatch) {
    return { 
        annotate: bindActionCreators(annotate, dispatch), 
        conceptEditors: bindActionCreators(conceptEditors, dispatch), 
        select_concept: bindActionCreators(select_concept, dispatch),
        highlight: bindActionCreators(highlight, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(Concept);
