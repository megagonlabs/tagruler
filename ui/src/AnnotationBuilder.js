import React from 'react';
import PropTypes from 'prop-types';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

import { annotate, select_link, TokenToRegex } from './actions/annotate'
import { spanAnnotate } from './actions/spanAnnotate'
import { conceptEditors, select_concept } from './actions/concepts'
import { label } from './actions/labelAndSuggestLF'
import AnnotationDisplay from './AnnotationDisplay'
import ConceptCollection from "./ConceptCollection";
import ClassLabelsCollection from "./ClassLabelsCollection"
import Span from './Span'
import ErrorSnackbar from './errorSnackbar'

import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import ClearIcon from '@material-ui/icons/Clear';
import Grid from '@material-ui/core/Grid';
import Tooltip from '@material-ui/core/Tooltip';
import { withTheme } from '@material-ui/core/styles'; // we use theme to pass color to svg elements


export const DIR_LINK = "Directed Link";
export const UNDIR_LINK = "Undirected Link";

export const DEFAULT_LABEL = 0;

class AnnotationBuilder extends React.Component {

    constructor(props) {
        super(props);
        this.elRef = React.createRef();
        this.default_link_indices = {[UNDIR_LINK]: 0, [DIR_LINK]: 100};

        this.state = {
            startOffset: 0,
            endOffset: 0,
            link_indices: this.default_link_indices
        };

        this.clickSegment = this.clickSegment.bind(this);
    }

    componentDidUpdate(nextProps) {
        if ((this.props.label !== null) && (this.props.entityPositions !== nextProps.entityPositions)) {
            //this.props.assignClassLabel(this.props.label);
        }
    }

    annotationFilteredString(root) {
        // iterate through the selected range to get a string representation
        // ignore annotation tags (these are meta-data and not part of the text to annotate)
        var _iterator = document.createTreeWalker(
            root,
            NodeFilter.SHOW_ALL,
            {
                acceptNode: function (node) {
                    if ((node.id==="annotation") || (node.parentNode.id==="annotation")) {
                        return NodeFilter.FILTER_REJECT;
                    } if (node.nodeType===3) {
                        return NodeFilter.FILTER_ACCEPT;
                    } else { 
                        return NodeFilter.FILTER_SKIP; 
                    }
                }
            }
        );

        var selection_string = ""
        var currentNode = _iterator.nextNode();
        while (currentNode) {
            selection_string += currentNode.textContent;
            currentNode = _iterator.nextNode()
        }
        return (selection_string)
    }

    setSelectedRange() {
        let start;
        let end;
        var sel = window.getSelection && window.getSelection();

        if (sel && sel.rangeCount > 0) {
            const range = window.getSelection().getRangeAt(0)
            // IT'S A TRAP: user is trying to highlight an annotation node!
            if (range.startContainer.parentNode && range.startContainer.parentNode.id === "annotation") {
                return (false);
            } else if (range.endContainer.parentNode && range.endContainer.parentNode.id === "annotation") {
                return (false);
            }
            // string containing the selected text
            const selection_str = this.annotationFilteredString(range.cloneContents());

            const preSelectionRange = range.cloneRange();
            preSelectionRange.selectNodeContents(this.elRef.current);
            preSelectionRange.setEnd(range.startContainer, range.startOffset);
            // string containing any text before the selection start
            const pre_sel_str = this.annotationFilteredString(preSelectionRange.cloneContents());
            start = pre_sel_str.length;
            end = start + selection_str.length;
        }

        // trim whitespace from selection
        while (/\s/.test(this.props.text[start]) && (start < end)){
            start++;
        }
        while (/\s/.test(this.props.text[end-1]) && (start < end)) {
            end--;
        }
        
        this.startOffset = start;
        this.endOffset = end;

        if (this.validRange()){ 
            this.addAnnotation(DEFAULT_LABEL); 
        }
    }


    validRange() {
        if (this.startOffset === this.endOffset) {
            return false;
        }
        if (this.startOffset > this.props.text.length || this.endOffset > this.props.text.length) {
            return false;
        }
        if (this.startOffset < 0 || this.endOffset < 0) {
            return false;
        }
        for (let i = 0; i < this.props.entityPositions.length; i++) {
            const e = this.props.entityPositions[i];
            if ((e.start_offset <= this.startOffset) && (this.startOffset < e.end_offset)) {
                return false;
            }
            if ((e.start_offset < this.endOffset) && (this.endOffset < e.end_offset)) {
                return false;
            }
            if ((this.startOffset < e.start_offset) && (e.start_offset < this.endOffset)) {
                return false;
            }
            if ((this.startOffset < e.end_offset) && (e.end_offset < this.endOffset)) {
                return false;
            }
        }
        return true;
    }

    addAnnotation(labelId=this.props.selectedConcept) {
        if (this.validRange()) {
            const newSpan = Span(this.startOffset, this.endOffset, this.props.text, labelId);
            const newEntityPositions = this.updatePositions([newSpan]);
            this.props.annotate(newEntityPositions);
        }
    }

    removeAnnotation(start_offset) {
        var link = null;
        var newEntityPositions = this.props.entityPositions.filter( 
            position => {
                if (position.start_offset===start_offset) {
                    link = position.link;
                    return false;
                }
                return true
            }
        )
        
        if (link !== null){
            newEntityPositions = newEntityPositions.filter( 
                position => {
                    if (position.link===link) {
                        position.link = null;
                    }
                    return true;
                }
            )
        }
        //TODO: implement deleting annotation correctly
        const data = {
            "label": 0, //TODO: assign label
            "annotations": newEntityPositions,
            "text": this.props.text,
            "index": this.props.index
        }
        this.props.annotate(newEntityPositions);
        this.props.spanAnnotate(newEntityPositions);
        this.props.label(data);

    }

    assignSpanLabel(label) {
        let newEntityPositions = this.props.entityPositions;
        let newAnnotations = [];
        for (let i = 0; i < newEntityPositions.length; i++) {
            const e = newEntityPositions[i];
            if (!e.spanAnnotated) {
                newEntityPositions[i].spanAnnotated = true;
                newEntityPositions[i].spanLabel = label;
                newAnnotations = [...newAnnotations, newEntityPositions[i]]
            }
        }
        const data = {
            "label": label,
            "annotations": this.props.annotations,
            "text": this.props.text,
            "index": this.props.index
        }
        this.props.spanAnnotate(newAnnotations);
        this.props.label(data);
    }

    assignToConcept(segment, label=this.props.selectedConcept) {
        var pos = {}
        if (segment.label === label) { //reset label
            pos = {...segment, label: DEFAULT_LABEL};
        } else {
            pos = {...segment, label: label};
        }
        const newEntityPositions = this.updatePositions([pos]);
        this.props.annotate(newEntityPositions);
    }

    addToConcept(segment) {
        if (this.props.selectedLink.type===null){
            const conceptName = this.props.selectedConcept;

            const view = this.conceptsRef.state.view;

            let tokens = this.props.concepts[conceptName].tokens;
            let text = this.props.text.slice(segment.start_offset, segment.end_offset)
            let new_span = text.trim();
            if (new_span !== "") {
                if (view===0) {
                    new_span = TokenToRegex(new_span);
                }
                const new_token = {string: new_span, type: view, case_sensitive: false}
                if (!(new_token in tokens)){
                    tokens.push(new_token);
                    this.props.conceptEditors.updateConcept(conceptName, tokens); 
                }
            }
        }
    }

    queueLink(segment, link_type) {
        this.props.select_link({type: link_type, segment: segment});
    }

    youCantDoubleLinkError() {
        this.setState({errorSnackbar: true});
        console.error(`Only one link is allowed per span.`);
    }

    addLink(segment) {
        const link_type = this.props.selectedLink.type;

        if (segment.link !== null) {
            // target segment already has a link!
            this.youCantDoubleLinkError();
            this.props.select_link({type: null});
            return false;
        } else if (link_type !== null) { // a link is queued
            if (this.props.selectedLink.segment !== segment) {

                let link_idx = this.state.link_indices[link_type];
                const pos1 = {...this.props.selectedLink.segment, link: link_idx};  
                if (link_type === DIR_LINK){
                    link_idx = -link_idx;
                }
                const pos2 = {...segment, link: link_idx};  
                const newEntityPositions = this.updatePositions([pos1, pos2]);
                this.setState({link_indices: {...this.state.link_indices, [link_type]: link_idx + 1}});
                this.props.annotate(newEntityPositions);
                this.props.select_link({type: null});

            } else {
                // pass. The same segment was selected twice.
            }
        } else {
            // no link is queued; queue this segment to be linked.
            // currently, links are always undirected
            this.queueLink(segment, UNDIR_LINK);
        }
    }
    

    removeLink(segment1, segment2=null) {
        if (segment2===null) {
            const link_id = segment1.link;
            const new_arr = this.props.entityPositions.filter(seg => (seg.link === link_id));
            if (new_arr.length !== 2) {
                console.error("Detected invalid link");
            }
            segment1 = new_arr[0];
            segment2 = new_arr[1];
        }
        segment1 = {...segment1, link: null};
        segment2 = {...segment2, link: null};
        let newEntityPositions = this.updatePositions([segment1, segment2]);
        this.props.annotate(newEntityPositions);
    }

    updatePositions(positionsToAdd){
        let newPositions = this.props.entityPositions;
        for (let index = 0; index < positionsToAdd.length; index++){
            let p = positionsToAdd[index];
            newPositions = newPositions.filter( 
            position => position.start_offset!==p.start_offset
            );
            newPositions.push(p);
        }
        newPositions.sort((a, b) => a.start_offset - b.start_offset);
        return(newPositions);
    }

    clickSegment(segment) {
        if (this.props.selectedLink.type !== null) {
            this.addLink(segment);
        }
    }

    clickLinkButton(segment) {
        if (segment.link !== null) {
            this.removeLink(segment);
        } else {
            this.addLink(segment);
        }
    }

    clearAllAnnotations() {
        this.props.annotate([]);
    }

    render(){
        const classes = this.props.classes;
        const annotations = this.props.entityPositions;

        return(
            <React.Fragment>
                <Grid item>
                        <Card className={classes.card} xs={12}>
                            <CardActions className={classes.cardActions}>
                                <Grid container direction="row" justify="space-evenly" alignItems="flex-end">
                                <ClassLabelsCollection classes={classes} onClick={this.assignSpanLabel.bind(this)}/>
                                </Grid>
                            </CardActions>
                            <CardContent >
                                <Grid container direction="row" alignItems="baseline" className={classes.grid} spacing={6}>
                                    {/*<Grid item s={3} xs={4}>
                                        <ConceptCollection childRef={(ref) => (this.conceptsRef = ref)} classes={classes} addAnnotations={this.updatePositions.bind(this)}/>
        </Grid>*/}
                                    <Grid container item direction="column" s={9} xs={12} spacing={6}>
                                        <Grid container direction="row" justify="flex-end">
                                            <Tooltip title="Clear all annotations" enterDelay={500}>
                                                <Button onClick={this.clearAllAnnotations.bind(this)}>
                                                    <ClearIcon/>
                                                </Button>
                                            </Tooltip>
                                        </Grid>
                                        <Grid item>
                                        <Box>
                                            <AnnotationDisplay 
                                                classes={classes} 
                                                onMouseUp={this.setSelectedRange.bind(this)} 
                                                textAreaRef={this.elRef}
                                                annotations={annotations}
                                                highlights={this.props.highlights}
                                                ners={this.props.ners}
                                                text={this.props.text}
                                                clickSegment={this.clickSegment.bind(this)}
                                                clickLinkButton={this.clickLinkButton.bind(this)}
                                                onDelete={this.removeAnnotation.bind(this)}
                                                />
                                        </Box></Grid>
                                    </Grid>
                                </Grid> 
                            </CardContent>
                        </Card>
                </Grid>

                <ErrorSnackbar open={this.state.errorSnackbar} setOpen={(bool) => this.setState({errorSnackbar: bool})} />
            </React.Fragment>
        );
    }
}

AnnotationBuilder.propTypes = {
  annotate: PropTypes.func.isRequired,
  assignClassLabel: PropTypes.func.isRequired,
  concepts: PropTypes.objectOf(PropTypes.object),
  entityPositions: PropTypes.array,
  highlights: PropTypes.array,
  selectedConcept: PropTypes.string,
  selectedLink: PropTypes.object,
  text: PropTypes.string.isRequired
};


function mapStateToProps(state, ownProps?) {
    return { 
        concepts: state.concepts.data, 
        index: state.text.index,
        entityPositions: state.annotations, 
        highlights: state.highlights.data,
        ners: state.ners,
        label: state.label,
        selectedConcept: state.selectedConcept,
        selectedLink: state.selectedLink,
        annotations: state.annotations,
        spanAnnotations: state.spanAnnotations,
    };
}

function mapDispatchToProps(dispatch) {
    return { 
        annotate: bindActionCreators(annotate, dispatch), 
        spanAnnotate: bindActionCreators(spanAnnotate, dispatch), 
        conceptEditors: bindActionCreators(conceptEditors, dispatch),
        select_link: bindActionCreators(select_link, dispatch),
        select_concept: bindActionCreators(select_concept, dispatch),
        label: bindActionCreators(label, dispatch),
    };
}

export default withTheme(connect(mapStateToProps, mapDispatchToProps)(AnnotationBuilder));