// based on https://github.com/rit-git/studio/blob/master/frontend/src/components/Datasets.js

import React, { Component } from 'react';
import { withRouter } from "react-router-dom";
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

// MUI components
import Button from '@material-ui/core/Button';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import TextField from '@material-ui/core/TextField';
import Typography from '@material-ui/core/Typography';

import { DropzoneDialog } from 'material-ui-dropzone'

// icons
import AddIcon from '@material-ui/icons/Add';

import api from './actions/api'
import UploadData, { fetchDatasets, setSelected } from './actions/datasets'



class Dataset extends Component {

  constructor(props) {
    super(props);


    this.state = {
      project: {},
      new_dataset: '',
      datafiles: [],
      dropDialogOpen: false,
    }

    //this.project_uuid = this.props.match.params.project_uuid;
    this.project_uuid = "SAMPLE_PROJ";
    //this.dataset_uuid = this.props.match.params.dataset_uuid;
    this.dataset_uuid = "SAMPLE_DATASET";


    this.fetchDatasets = this.fetchDatasets.bind(this);
    this.fetchDatasetDatafiles = this.fetchDatasetDatafiles.bind(this);

    this.handleBreadcrumbProjectClick = this.handleBreadcrumbProjectClick.bind(this);

    this.handleDropDialogOpen = this.handleDropDialogOpen.bind(this);
    this.handleDropDialogClose = this.handleDropDialogClose.bind(this);
    this.handleDropFileSave = this.handleDropFileSave.bind(this);

    this.handleDataFileAdd = this.handleDataFileAdd.bind(this);
    this.handleDataFileOpen = this.handleDataFileOpen.bind(this);
    this.handleDataFileRemove = this.handleDataFileRemove.bind(this);
    this.handleDataFileRefresh = this.handleDataFileRefresh.bind(this);

    this.handleInput = this.handleInput.bind(this);
    this.handleDataSelect = this.handleDataSelect.bind(this);
  }

  componentDidMount() {
    this.fetchDatasets();

    /*this.projectIntervalTimer = setInterval(() => {
      this.fetchProject()
    }, 3000);

    this.fetchDataset();

    this.intervalTimer = setInterval(() => {
      this.fetchDatasets()
    }, 3000);

    this.fetchDatasetDatafiles();

    this.intervalDatafilesTimer = setInterval(() => {
      this.fetchDatasetDatafiles()
    }, 3000);*/

  }

  handleInput(event) {
    this.setState({
      new_dataset: event.target.value
    });
  }

  handleDataSelect(event) {
    const { options } = event.target;
    const value = [];
    for (let i = 0, l = options.length; i < l; i += 1) {
      if (options[i].selected) {
        value.push(options[i].value);
      }
    }
    this.props.setSelected(value);
  }

  fetchDatasets() {
    if ( !this.props.user )
      return;
    this.props.user.getIdToken(true).then((idToken) => {
      this.props.fetchDatasets(idToken)
    });
  }

  fetchDatasetDatafiles() {

    if ( !this.props.user )
      return;

    let dataset_uuid = this.dataset_uuid;

    this.props.user.getIdToken(true).then((idToken) => {

      fetch(api + '/datasets/datafiles/' + dataset_uuid,
        {
          method: 'GET',
          headers: {
            'Authorization': 'Bearer ' + idToken
          }
        })
        .then(res => res.json())
        .then((data) => {
          this.setState({'datafiles': data});
          clearInterval(this.intervalDatafilesTimer);
        })
      .catch(console.log)
      // ...
    }).catch(function(error) {
      // Handle error
    });
  }

  addDataFile(file) {
    if ( !this.props.user )
      return;

    var form = new FormData();
    form.append('file', file)
    let dataset_uuid = this.state.new_dataset;

    this.props.user.getIdToken(true).then((idToken) => {
      this.props.UploadData(form, 
        dataset_uuid,
        idToken,
      );
    });
  }

  removeDataFile(datafile) {
    if ( !this.props.user )
      return;

    let datafile_uuid = datafile.uuid;
    let dataset_uuid = this.dataset_uuid;

    this.props.user.getIdToken(true).then((idToken) => {

      fetch(api + '/datasets/datafiles/' + dataset_uuid + '/' + datafile_uuid,
        {
          method: 'DELETE',
          headers: {
            'Authorization': 'Bearer ' + idToken
          }
        })
        .then(res => res.json())
        .then((data) => {
          this.fetchDatasetDatafiles();
        })
      .catch(console.log)
      // ...
    }).catch(function(error) {
      // Handle error
    });
  }


  handleDataFileRefresh() {
    this.fetchDatasetDatafiles();
  }

  handleDataFileAdd(files) {
    for ( let fi in files ) {
      this.addDataFile(files[fi]);
    }
    this.fetchDatasets();
  }

  handleDataFileRemove(datafile) {
    this.removeDataFile(datafile);
  }

  handleDropDialogClose() {
    this.setState({
        dropDialogOpen: false
    });
    this.fetchDatasets();
  }


  handleDropFileSave(files) {
    this.handleDataFileAdd(files);
    this.setState({
        dropDialogOpen: false,
        selected_dataset: this.state.new_dataset
    });
  }

  handleDropDialogOpen() {
    
    this.setState({
        dropDialogOpen: true
    });

  }

  handleDataFileOpen(datafile) {
    let project_uuid = this.project_uuid;
    let dataset_uuid = this.dataset_uuid;
    if ( project_uuid ) {
      this.props.history.push("/projects/datasets/datafiles/" + project_uuid + "/" + dataset_uuid + "/" + datafile.uuid);
    }
    else
      this.props.history.push("/datasets/datafiles/" + dataset_uuid + "/" + datafile.uuid);
  }


  handleBreadcrumbProjectClick() {
    this.props.history.push("/projects/" + this.state.project.uuid);
  }

  render() {
    const { classes } = this.props;
    const dataset_names = this.props.datasets;

    return (
      <div>
        <Typography>To begin, select the dataset you wish to label.</Typography>
        <div className={classes.contents}>
          <FormControl className={classes.formControl}>
            <Select multiple native value={this.props.selected_dataset} onChange={this.handleDataSelect}>
              { dataset_names.map((dataset) => 
                <option 
                  value={dataset} 
                  key={dataset}
                >
                  {dataset}
                </option>) }
            </Select>
          </FormControl>
        </div>
        
        <Typography>
          Alternatively, upload your own dataset. (Currently only csv files are supported). <br/>
          You will need a column named 'text' and a column named 'label' (which is set to NaN if the label is unknown).
        </Typography>
          <div className={classes.contents}><FormControl className={classes.formControl}>

            <TextField 
              placeholder="new dataset name"
              onChange = {this.handleInput}
              value={this.state.new_dataset}
              InputProps={{
                endAdornment: 
                  <Button onClick={this.handleDropDialogOpen} disabled={(this.state.new_dataset.trim()==="")}>
                    <AddIcon/>
                  </Button>,
              }}/>

            <DropzoneDialog
                open={this.state.dropDialogOpen}
                dialogTitle="Upload CSV file"
                dropzoneText="Drag and drop a CSV file or click to upload"
                onSave={this.handleDropFileSave.bind(this)}
                acceptedFiles={['text/csv']}
                showPreviews={true}
                useChipsForPreview={true}
                maxFileSize={20000000}
                onClose={this.handleDropDialogClose.bind(this)}
            />
        </FormControl></div>
      </div>
    )
  }
}

function mapStateToProps(state, ownProps?) {
    return {
      datasets: state.datasets.data,
      selected_dataset: state.datasets.selected
    };
}
function mapDispatchToProps(dispatch) {
    return { 
        UploadData: bindActionCreators(UploadData, dispatch),
        fetchDatasets: bindActionCreators(fetchDatasets, dispatch),
        setSelected: bindActionCreators(setSelected, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(withRouter(Dataset));