// material ui
import Box from '@material-ui/core/Box';
import Button from '@material-ui/core/Button';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import Step from '@material-ui/core/Step';
import StepContent from '@material-ui/core/StepContent';
import StepLabel from '@material-ui/core/StepLabel';
import Stepper from '@material-ui/core/Stepper';
import Typography from '@material-ui/core/Typography';

import clsx from 'clsx';
import React, {useRef, useEffect} from 'react';
import {connect} from "react-redux";
import { Link as RouteLink } from "react-router-dom";
import { Redirect } from "react-router-dom";
import {bindActionCreators} from 'redux';

// actions
import {submitLabelsAndName} from './actions/labelClasses';
import launch, { launchStatus } from './actions/loadingBar';
import fetchClasses, {addLabelClass} from './actions/labelClasses';

// components
import Dataset from './Dataset';
import LabelCreation from './LabelCreation';
import { useStyles } from './ProjectGrid';

function LinearProgressWithLabel(props) {
  return (
    <Box display="flex" alignItems="center">
      <Box width="100%" mr={1}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
      <Box minWidth={35}>
        <Typography variant="body2" color="textSecondary">{`${Math.round(
          props.value,
        )}%`}</Typography>
      </Box>
    </Box>
  );
}

function VerticalLinearStepper(props) {

    const [activeStep, setActiveStep] = React.useState(0);
    const steps = getSteps();

    const classes = useStyles(),
        isDrawerOpen = props.isDrawerOpen;

    // this is used for getIdToken for a dummy user object
    const promise1 = new Promise((resolve, reject) => {
        resolve(true);
    });

    const handleNext = () => {
        //TODO If we're on the label creation step, send the labels to server when next is clicked
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleBack = () => {
        setActiveStep((prevActiveStep) => prevActiveStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };

    function getSteps() {
      return ['Select Dataset', 'Create Project'];
    }

    function getStepContent(step) {
      switch (step) {
        case 0:
          return (  <Dataset isDrawerOpen={isDrawerOpen} 
                        user={{getIdToken: (arg) => promise1}}
                        classes={classes}
                    />);
        case 1:
          return (<LabelCreation classes={classes}/>);
        case 2:
          return (<RouteLink to="/label" classes={classes}>Continue to Task</RouteLink>);
        default:
          return 'Unknown step';
      }
    }

    function getStepButton(step) {

        switch (step) {
            case 1:
              function goToProject() {
                props.submitLabelsAndName(props.labelClasses);
                props.launch();
                props.fetchClasses();
              }
              return (<Button
                          variant="contained"
                          color="primary"
                          onClick={goToProject}
                          className={classes.button}
                          disabled={props.labelClasses.length===0}
                      >
                          Create Project
                      </Button>)
            default:
              return (<Button
                          variant="contained"
                          color="primary"
                          onClick={handleNext}
                          className={classes.button}
                          disabled={props.selected_dataset===undefined}
                      >
                          Next
                      </Button>)
        }
    }


    const intervalRef = useRef();

    useEffect(() => {
      const id = setInterval(() => {
        if (props.inProgress){
          props.getLaunchProgress(props.launchThread);
        }      
      }, 1000);
      intervalRef.current = id;
      return () => {
        clearInterval(intervalRef.current);
      };
    });
    
    if (props.launchProgress >= 100) {
      return ( <Redirect to="/project" />)
    }

    if (props.inProgress) {
      return(
          <div id="loadingBar">
            <Typography> Your project is loading. You will be automatically redirected when it is complete.</Typography>
            <br /> 
            <LinearProgressWithLabel variant="determinate" value={props.launchProgress}/>
          </div>
        )
    }

    return (
        <div className={clsx(classes.content, { [classes.contentShift]: isDrawerOpen })}>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
                <StepContent>
                  {getStepContent(index)}
                  <div className={classes.actionsContainer}>
                    <div>
                      <Button
                        disabled={activeStep === 0}
                        onClick={handleBack}
                        className={classes.button}
                      >
                        Back
                      </Button>
                      {getStepButton(index)}
                    </div>
                  </div>
                </StepContent>
              </Step>
            ))}
          </Stepper>
          {activeStep === steps.length && (
            <Paper square elevation={0} className={classes.resetContainer}>
              <Typography>All steps completed - you&apos;re finished</Typography>
              <Button onClick={handleReset} className={classes.button}>
                Reset
              </Button>
            </Paper>
          )}
        </div>
    );
}

function mapStateToProps(state, ownProps?) {
    return {
      selected_dataset: state.datasets.selected,
      labelClasses: state.labelClasses.data,
      launchProgress: state.launchProgress.progress*100, //convert to a percentage, not a fraction
      inProgress: (!(state.launchProgress.thread === null)),
      launchThread: state.launchProgress.thread,
    };
}
function mapDispatchToProps(dispatch) {
    return { 
      submitLabelsAndName: bindActionCreators(submitLabelsAndName, dispatch),
      launch: bindActionCreators(launch, dispatch),
      getLaunchProgress: bindActionCreators(launchStatus, dispatch),
      fetchClasses: bindActionCreators(fetchClasses, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(VerticalLinearStepper);