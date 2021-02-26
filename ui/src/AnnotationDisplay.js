import React from 'react';
import {connect} from "react-redux";
import PropTypes from 'prop-types';

import Link from './Link'
import SelectedSpan from './SelectedSpan'
import Span from './Span'
import Typography from '@material-ui/core/Typography';

import { NERSpan, MatchedSpan} from './RichTextUtils'

export const DEFAULT_LABEL = 0;
// TODO use defaultdict instead of default label?

class AnnotationDisplay extends React.Component {


    addRichText(chunks, highlights) {
        var h_index = 0;
        var highlight = highlights[h_index];
        var rich_texts = [];
        for (var chunk of chunks) {
            var text = [];
            var loc = 0;
            while ((h_index < highlights.length) && (chunk.end_offset > highlight.start_offset)) {
                var bold_start = Math.max(0, highlight.start_offset - chunk.start_offset);
                if (bold_start > 1) {
                    text.push(<React.Fragment key={loc}>{chunk.text.slice(loc, bold_start)}</React.Fragment>);
                }
                var bold_end = Math.min(0, highlight.end_offset - chunk.end_offset) + chunk.text.length;
                text.push(MatchedSpan(chunk.text.slice(bold_start, bold_end), highlight, bold_start));
                loc = bold_end;
                h_index++;
                highlight = highlights[h_index];
            }
            if (chunk.text.length > loc) {
                text.push(<React.Fragment key={loc}>{chunk.text.slice(loc, chunk.text.length)}</React.Fragment>);
            }
            chunk.rich_text = text.join("");
            rich_texts.push(text)
        }
        return rich_texts;
    }

    addNER(chunks, NERs) {
        var h_index = 0;
        var highlight = NERs[h_index];
        var rich_texts = [];
        for (var chunk of chunks) {
            var text = [];
            var loc = 0;
            while ((h_index < NERs.length) && (chunk.end_offset > highlight.start_offset)) {
                var bold_start = Math.max(0, highlight.start_offset - chunk.start_offset);
                if (bold_start > 1) {
                    text.push(<React.Fragment key={loc}>{chunk.text.slice(loc, bold_start)}</React.Fragment>);
                }
                var bold_end = Math.min(0, highlight.end_offset - chunk.end_offset) + chunk.text.length;
                text.push(NERSpan(chunk.text.slice(bold_start, bold_end), highlight, bold_start));
                loc = bold_end;
                h_index++;
                highlight = NERs[h_index];
            }
            if (chunk.text.length > loc) {
                text.push(<React.Fragment key={loc}>{chunk.text.slice(loc, chunk.text.length)}</React.Fragment>);
            }
            rich_texts.push(text);
        }
        return rich_texts;
    }

    chunks_old(annotations, text) {
        const res = [];
        let left = 0;
        for (let i = 0; i < annotations.length; i++) {
            const e = annotations[i];
            // annotations must be sorted by start offset, and must not overlap!
            if (e.start_offset < left){
                console.error("Annotations are overlapping or out of order.");
                console.error(annotations);
            }

            if (left !== e.start_offset){
                const l = Span(left, e.start_offset, text);
                res.push(l); 
            }
            e.text = text.slice(e.start_offset, e.end_offset);
            res.push(e);
            left = e.end_offset;
        }
        const l = Span(left, text.length, text);
        res.push(l);
        return res;
    }

    createSelectedSpan(chunk, rich_text=null, divStyle={}) {
        return (
            <SelectedSpan 
                key={chunk.annotation.start_offset} 
                id={ `selection_${chunk.annotation.start_offset}` }
                classes={this.props.classes} 
                style={divStyle}
                text={rich_text ? rich_text : chunk.text}
                linked={chunk.annotation.link !== null}
                sid={chunk.annotation.start_offset} 

                // optional props
                clickSegment = {"clickSegment" in this.props ? () => this.props.clickSegment(chunk.annotation) : undefined}
                clickLinkButton = {"clickLinkButton" in this.props ? () => this.props.clickLinkButton(chunk.annotation) : undefined }
                onDelete = {"onDelete" in this.props ? () => this.props.onDelete(chunk.annotation.start_offset) : undefined } 
            />);
    }

    getStyle() {     // returns a dictionary of styling information for each concept
        const default_style = {
            id: DEFAULT_LABEL, 
            text_color: 'inherit', 
            background_color: '#D3D3D3'
        };
        // define styling for highlighted spans without assigned concepts (id=DEFAULT_LABEL)
        var getStyle = new Proxy({}, {
          get: (target, name) => name in target ? target[name] : default_style
        })
        const concepts = Object.keys(this.props.concepts);
        for (let i = 0; i < concepts.length; i++) {
            let concept = concepts[i];
            getStyle[concept] = { 
                id: concept,
                text_color: "white", 
                background_color: this.props.concepts[concept]["color"],
            };
        }
        return getStyle;
    }
    
    chunks(offsets, text, selection_arrays) {
        var uniqueAndSorted = [...new Set(offsets)].sort((a, b) => a-b);
        const res = [];
        let left = 0;
        for (let offset of uniqueAndSorted) {
            // annotations must be sorted by start offset, and must not overlap!
            var l = Span(left, offset, text);
            for (var selection_array of selection_arrays) {
                var selection = this.isItContained(offset, selection_array)
                if (selection) {
                    l[selection_array[0].type] = selection;
                    //TODO: do this more efficiently
                    l["label"] = selection_array[0].label;
                } 
            }
            res.push(l); 
            left = offset;
        }
        return res;
    }

    isItContained(offset, selection_array) {
        for (var selection of selection_array) {
            if (offset > selection.start_offset) {
                if (offset <= selection.end_offset) {
                    return (selection);
                }
            }
        } return (false);
    }

    formatRichText(chunk) {
        var text = chunk.text;
        if (chunk.text_match) {
            text = MatchedSpan(text, chunk.text_match, chunk.start_offset)
        }
        if (chunk.named_entity) {
            text = NERSpan(text, chunk.named_entity, chunk.start_offset);
        }
        return [text];
    }
    render() {
        const getStyle = this.getStyle();


        const classes = this.props.classes;
        let annotations = this.props.annotations || [];
        annotations.map(a => a.type = "annotation");
        let highlights = this.props.highlights || [];
        highlights.map(a => a.type = "text_match");
        let ners = this.props.ners || [];
        ners.map(a => a.type = "named_entity");

        let offsets = [annotations, highlights, ners].flat().map(a => [a.start_offset, a.end_offset]).flat()
        offsets.push(this.props.text.length);
        const chunks = this.chunks(offsets, this.props.text, [annotations, highlights, ners]);

        let links = {};
        let tags = [];

        const labelClasses = this.props.labelClasses;
        for (var i = 0; i < chunks.length; i++) {
            var chunk = chunks[i];
            let text = this.formatRichText(chunk);
            if (chunk.annotation) {
                let divStyle = { 
                    color: (chunk.annotation.spanAnnotated)? labelClasses[chunk.annotation.spanLabel].textColor:getStyle[chunk.annotation.label].text_color,
                    backgroundColor: (chunk.annotation.spanAnnotated)? labelClasses[chunk.annotation.spanLabel].backgroundColor:getStyle[chunk.annotation.label].background_color,
                };
                if (('link' in chunk.annotation) && (chunk.annotation.link !== null)) {
                    if (!(chunk.annotation.link in links)) {
                        links[chunk.annotation.link] = [];
                    }
                    links[chunk.annotation.link].push(chunk);
                }
                while ((i + 1 < chunks.length) && (chunks[i+1].annotation) && (chunks[i+1].annotation.id === chunk.annotation.id)) {
                    i++;
                    var new_chunk = chunks[i];
                    text.push(this.formatRichText(new_chunk));
                }
                text = this.createSelectedSpan(chunk, text, divStyle);
            }

            tags.push(
            <Typography 
                component="div"
                className={classes.text}
                key={chunk.start_offset+'_'+chunk.end_offset}
            >{text}</Typography> )
        }

        links = Object.values(links).map( link => {
            const from=`selection_${link[0].id}`;
            const to=`selection_${link[1].id}`;
            return(<Link 
                key={["link", "from", from, "to", to].join("_")} 
                from={from} 
                to={to}
                classes={classes}
                className={classes.link}
            />);
        });


        return (
        <React.Fragment>
        <div className={this.props.classes.text} 
            id='text-to-label' 
            onMouseUp={this.props.onMouseUp}
            ref={this.props.textAreaRef}>
            {tags}
            <br/>
        </div>
        {links}
        </React.Fragment>
    )}
}

AnnotationDisplay.propTypes = {
    text: PropTypes.string.isRequired,
    annotations: PropTypes.array,
    highlights: PropTypes.array,
    ners: PropTypes.array,
    onMouseUp: PropTypes.func,
    classes: PropTypes.object,
};

function mapStateToProps(state, ownProps?) {
    return { 
        concepts: state.concepts.data,
        labelClasses: state.labelClasses.data
    }
}
function mapDispatchToProps(dispatch) {
    return {}
}

export default connect(mapStateToProps, mapDispatchToProps)(AnnotationDisplay)