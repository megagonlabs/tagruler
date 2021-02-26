import React from 'react';
import PropTypes from 'prop-types';
import {conceptEditors, select_concept} from './actions/concepts'
import { highlight } from './actions/annotate'
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

import AddIcon from '@material-ui/icons/Add';
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import Grid from '@material-ui/core/Grid';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import Typography from '@material-ui/core/Typography';

import Concept from './Concept'

import { TOKEN_VIEW, REGEX_VIEW } from './ConceptElement'


import TextField from '@material-ui/core/TextField';


class ConceptCollection extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      conceptName: "",
      fetched: false,
      view: TOKEN_VIEW, //view 0 is token view, view 1 is regex view
      openConcept: null,
    };
    this.handleInput = this.handleInput.bind(this);
    this.handleKeyPress = this.handleKeyPress.bind(this);
    this.handleAdd = this.handleAdd.bind(this);
    this.changeView = this.changeView.bind(this);
    this.openConcept = this.openConcept.bind(this);
    this.closeConcept = this.closeConcept.bind(this);

  }

  componentDidMount() {
      if (!this.state.fetched) {
        this.props.conceptEditors.fetchConcepts();
        this.setState({fetched: true});
      }
      const childRef = this.props.childRef;
      childRef(this);
  }
  componentWillUnmount() {
      const childRef = this.props.childRef;
      childRef(undefined);
  }

  openConcept(cname) {
    this.resetHighlights();
    this.setState({openConcept: cname});
  }

  closeConcept() {
    this.resetHighlights();
    this.setState({openConcept: null});
  }

  resetHighlights() {
      this.props.highlight([]);
  }

  changeView(new_view) {
      this.setState({...this.state, view: new_view})
  }

  handleInput(event) {
    this.setState({
      conceptName: event.target.value
    });
  }

  handleAdd() {
    const newConcept = this.state.conceptName.trim()
    if (newConcept !== "") {
      this.setState({conceptName: ""});
      this.props.conceptEditors.addConcept(newConcept);
      this.props.select_concept(newConcept);
      this.openConcept(newConcept);
    }
  }

  handleKeyPress(event){ 
      if (event.key === 'Enter') {
          this.handleAdd();
      }
  }

  render() {
    const conceptNames = Object.keys(this.props.concepts);
    const classes = this.props.classes;

    return (
      <Box> 
      <Grid container direction="row" justify="space-between">
          <Typography className={classes.title} variant="h6">Concepts</Typography>
          <Select onChange={(event) => this.changeView(event.target.value)} value={this.state.view} >
              <MenuItem children="Token View" value={TOKEN_VIEW}/>
              <MenuItem children="Regex View" value={REGEX_VIEW}/>
          </Select>
      </Grid>
      <br/>
      <Grid container direction="column" justify="flex-start" alignItems="stretch">
           { conceptNames.map( (concept) =>
                 <Grid item key={concept}><Concept open={() => this.openConcept(concept)} close={this.closeConcept} isOpen={this.state.openConcept===concept} mustClose={this.props.closeAll} view={this.state.view} key={concept} concept={concept} classes={classes} addAnnotations={this.props.addAnnotations}/></Grid>
              )
            }
          <Grid item>
            <TextField
               fullWidth={true}
               variant="outlined"
               className={classes.input}
               placeholder="add new concept"
               inputProps={{ 'aria-label': ' ' }}
               onKeyPress = {this.handleKeyPress}
               onChange = {this.handleInput}
               value={this.state.conceptName}
               InputProps={{endAdornment:
                <Button className={classes.button} size="small" color="primary" aria-label="add" 
                    onClick={this.handleAdd} disabled={(this.state.conceptName==="")} >
                    <AddIcon />
                </Button>}}
            />
          </Grid>
       </Grid>
       </Box>
    );
  }
}

ConceptCollection.propTypes = {
  addAnnotations: PropTypes.func,
  classes: PropTypes.object.isRequired,
  conceptEditors: PropTypes.object.isRequired
};

function mapStateToProps(state, ownProps?) {
    return { concepts: state.concepts.data };
}

function mapDispatchToProps(dispatch) {
    return { 
      conceptEditors: bindActionCreators(conceptEditors, dispatch),
      select_concept: bindActionCreators(select_concept, dispatch) ,
      highlight: bindActionCreators(highlight, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(ConceptCollection);
