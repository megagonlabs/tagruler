import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

import Button from '@material-ui/core/Button';
import CircularProgress from '@material-ui/core/CircularProgress';
import Container from '@material-ui/core/Container';
import Divider from '@material-ui/core/Divider';
import Grid from '@material-ui/core/Grid';
import IconButton from '@material-ui/core/IconButton';
import InfoIcon from '@material-ui/icons/Info';
import Paper from '@material-ui/core/Paper';
import Popover from '@material-ui/core/Popover';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';
import WarningIcon from '@material-ui/icons/Warning';

import AnnotationDisplayCollapse from './AnnotationDisplayCollapse'
import { style } from './SortingTableUtils'
import { getLFexamples } from './actions/submitLFs'
import { getInteraction, deleteInteraction, setAsCurrentInteraction } from './actions/interaction'

class LFPanel extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            anchorEl: null,
        };
    }

    handleOpen(event) {
        this.props.getLFexamples(this.props.lf.id);
        this.setState({anchorEl: event.currentTarget});
    }

    handleClose() {
        this.setState({
            anchorEl: null,
        });
        this.props.getLFexamples(null);
    }

    componentDidUpdate(prevProps) {
        if (prevProps.pending && !this.props.pending) {
            this.setState({pending: false});
        }
    }

    returnToInteraction() {
        this.props.setAsCurrentInteraction(this.props.lf.interaction_idx);
    }

    render() { 
        var classes = this.props.classes;
        const anchorEl = this.state.anchorEl;
        const open = (Boolean(anchorEl) && !this.props.pending);

        var fields_to_display = ["conditions", "label", "Coverage Training", "Conflicts Training", "Emp. Acc.", "Recall","Correct", "Incorrect"]
        var names_to_display =  ["Conditions", "Label", "Train Set Coverage", "Train Set Conflicts", "Precision", "Recall","Correct", "Incorrect", "Context"]

        if (this.props.lf.context) {
            fields_to_display.splice(0, "Context", 1);
            names_to_display.splice(0, "Context", 1);
        }
        return (
            <React.Fragment>
                <IconButton onClick={this.handleOpen.bind(this)} size="small"><InfoIcon/></IconButton>

                <Popover
                    aria-labelledby="Labeling Function Details"
                    open={open}
                    onClose={this.handleClose.bind(this)}
                    anchorEl={anchorEl}
                    marginThreshold={50}
                    style={{
                        maxHeight: "80vh",
                    }}
                >
                <Container>
                    <Grid
                        className={classes.grid}
                        container
                        direction="row"
                        justify="space-around"
                        style={{
                            marginTop: "50px",
                        }}                      
                        spacing={6}
                    >
                    
                        <Grid container item xs={6} className={classes.grid} spacing={5} padding={10}>
                        <Paper className={classes.paper}>
                            <Typography variant="h6">Statistics</Typography>
                            <Typography variant="body1">Computed over the training data unless otherwise noted.</Typography>
                            <Table size="small" aria-label="Statistics for this LF"><TableBody>
                                {fields_to_display.map((key, idx) => 
                                    <TableRow key={key}>
                                        <TableCell ><Typography variant="body2">{names_to_display[idx]}</Typography></TableCell> 
                                        <TableCell padding="none" align="right">{isNaN(this.props.lf[key]) ? this.props.lf[key] : style(this.props.lf[key])}</TableCell>
                                    </TableRow>
                                )}
                            </TableBody></Table>    
                            <br/>
                            <Button align="right" variant="contained" onClick={this.returnToInteraction.bind(this)}>Return to this interaction</Button>
                            <Typography variant="h6"> False Positives ({this.props.mistakes.length})</Typography>
                            <Typography variant="body1">Up to 5 examples from the development set that are INCORRECTLY labeled by this function. </Typography>
                            <br/>
                            {this.props.pending ? < CircularProgress/> : this.props.mistakes.map((example_dict, idx) => 
                                <Grid item key={idx}>
                                <Divider/>
                                <WarningIcon color={"error"}/>
                                    <AnnotationDisplayCollapse 
                                        text={example_dict.text || ""}
                                        annotations={example_dict.annotations || []}
                                        classes={classes}
                                    />
                                </Grid>
                                )}
                        </Paper>
                        </Grid>
                        <Grid container item xs={6} className={classes.grid} direction={'column'} justify={'flex-start'}>
                            <Paper className={classes.paper}>
                                <Typography variant="h6"> Matches ({this.props.examples.length})</Typography>
                                <Typography variant="body1">Examples from the training data that are labeled by this function. </Typography>
                                <br/>
                                {this.props.pending ? < CircularProgress/> : this.props.examples.map((example_dict, idx) => 
                                    <Grid item xs={12} key={idx}>
                                    <Divider/><Divider/>
                                    <AnnotationDisplayCollapse 
                                        text={example_dict.text || ""}
                                        annotations={example_dict.annotations || []}
                                        classes={classes}
                                    /></Grid>
                                )}
                            </Paper>
                        </Grid>
                    
                    </Grid>
                </Container>
                </Popover>
            </React.Fragment>
        );
    }
}

function mapStateToProps(state, ownProps?) {
    return {
        pending: !Boolean(state.labelExamples.examples), //TODO add pending field to labelExamples
        examples: state.labelExamples.examples || [],
        mistakes: state.labelExamples.mistakes || []
    }
}

function mapDispatchToProps(dispatch) {
    return {
        getInteraction: bindActionCreators(getInteraction, dispatch),
        deleteInteraction: bindActionCreators(deleteInteraction, dispatch),
        setAsCurrentInteraction: bindActionCreators(setAsCurrentInteraction, dispatch),
        getLFexamples: bindActionCreators(getLFexamples, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(LFPanel);