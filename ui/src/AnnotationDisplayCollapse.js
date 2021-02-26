import React from 'react';
import PropTypes from 'prop-types';

import AnnotationDisplay from './AnnotationDisplay'
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpandLessIcon from '@material-ui/icons/ExpandLess';
import Grid from '@material-ui/core/Grid';

class AnnotationDisplayCollapse extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            open: false
        }
        this.collapseText(this.props.text);
    }

    collapseText(text, MAX_COLLAPSED_MARGIN = 100) {
        const max_start = this.props.annotations[0].start_offset || 0;
        const min_end = this.props.annotations[0].end_offset || 0;

        var coll_annotations = this.props.annotations;
        if (max_start > MAX_COLLAPSED_MARGIN) {
            const start = text.indexOf(" ", max_start - MAX_COLLAPSED_MARGIN);
            if (start > 30) {
                coll_annotations = this.props.annotations.map((ann) => {
                    return ({
                        ...ann,  
                        start_offset: ann.start_offset - start + 3,                
                        end_offset: ann.end_offset - start + 3
                    })
                })
                text = "..." + text.slice(start);
            }
        }

        if (min_end < text.length - MAX_COLLAPSED_MARGIN) {
            const end = text.lastIndexOf(" ", min_end + MAX_COLLAPSED_MARGIN);
            if (text.length - end > 30){
                text = text.slice(0, end) + "...";
            }
        }

        this.coll_text = text;
        this.coll_annotations = coll_annotations;
    }

    toggleOpen() {
        this.setState({
            open: !this.state.open
        })
    }

    toggleIcon() {
        if (this.state.open) {
            return <ExpandLessIcon onClick={this.toggleOpen.bind(this)}/>
        } else {
            return <ExpandMoreIcon onClick={this.toggleOpen.bind(this)}/>
        }
        
    }

    render() {

        var collapsedText = this.coll_text;
        var coll_annotations = this.coll_annotations;

        return(
            <Grid container item
                direction="row" 
                xs={12}
                justify="space-around"
            >
                <Grid item xs={11}><AnnotationDisplay 
                    text={this.state.open ? this.props.text : collapsedText}
                    annotations={this.state.open ? this.props.annotations : coll_annotations}
                    classes={this.props.classes}
                /></Grid>
                <Grid item xs={1}>{ this.props.text !== collapsedText ? this.toggleIcon() : null}</Grid>
            </Grid>
        )
    }
}

AnnotationDisplayCollapse.propTypes = {
    text: PropTypes.string,
    annotations: PropTypes.array
}

export default AnnotationDisplayCollapse