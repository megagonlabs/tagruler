import React from 'react';
import Button from '@material-ui/core/Button';
import {connect} from "react-redux";
import {bindActionCreators} from 'redux';

import ArrowDropUpIcon from '@material-ui/icons/ArrowDropUp';
import ArrowDropDownIcon from '@material-ui/icons/ArrowDropDown';
import Divider from '@material-ui/core/Divider';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import RefreshIcon from '@material-ui/icons/Refresh';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';

import { getLRStatistics } from './actions/getStatistics'
import { style } from './SortingTableUtils'


class CRFStatisticsPane extends React.Component {
    componentDidUpdate(prevProps, prevState) {
      if ((prevProps.statistics !== this.props.statistics) && (Object.keys(prevProps.statistics).length > 0)) {
        this.setState({prevStats: prevProps.statistics});
      }
    }

    statDelta(key) {
      if ((this.state) && ("prevStats" in this.state)) {
        const delta = this.props.statistics[key] - this.state.prevStats[key];
        let cellContent = delta;
        if (delta > 0.0){
          cellContent = <span style={{color:"green"}}><ArrowDropUpIcon/> {style(delta)} </span>
        } else if (delta < -0.0) {
          cellContent = <span style={{color:"red"}}><ArrowDropDownIcon/> {style(delta)}</span>
        } 
        return(<TableCell>{cellContent}</TableCell>)
      }
    }

    render(){
        const classes = this.props.classes;
        const prevStats = ((this.state) && ("prevStats" in this.state));
        const labelKeyMap = {
          'Precision': 'Precision',
          'Recall': 'Recall',
          'F1-score': 'F1-score'
        }
        const labelClasses = this.props.labelClasses
        return(
          <Paper className={classes.paper}>
            
            <Typography className={classes.title} variant="h6" id="tableTitle">
                <RefreshIcon onClick={this.props.getLRStatistics} disabled={this.props.pending}></RefreshIcon>Trained Model Statistics
            </Typography>
            <Typography variant="body1">Train a conditional random field (CRF) model on your training set.</Typography>

            {this.props.pending ? <LinearProgress/> : <Divider/>}

            <Table stickyHeader className={classes.table} size="small" aria-label="labeling statistics">
              <TableBody>
                {["Precision","Recall", "F1-score"].map(key => { 
                  if (key in this.props.statistics) {
                      return (<TableRow key={key}>
                          <TableCell>{labelKeyMap[key]}</TableCell>
                          <TableCell align="right">{style(this.props.statistics[key])}</TableCell>
                          {this.statDelta(key)}
                        </TableRow>)
                  } else { return null}
                })}</TableBody>
                </Table>
            <br/>
            <Typography className={classes.title} variant="h6" id="tableTitle">
                Class-Specific Statistics
            </Typography>
                <Table>
                <TableBody>
                  <TableRow key="class_headers">
                    <TableCell>
                    </TableCell>
                    {this.props.labelClasses.map(label => (
                      prevStats ?
                        [label.name !== "O" ? <TableCell align="right">
                          <Button 
                            className={classes.button + label.name} 
                            key = {label.key}
                            style={
                            {backgroundColor:label.backgroundColor,
                            color:label.textColor,
                            fontWeight:'bold',
                            margin:'5px',
                            }} 
                            size={"small"}
                          >
                            {label.name}
                          </Button>
                        </TableCell> : null,
                        <TableCell>
                      </TableCell>]: [label.name !== "O" ? <TableCell align="right">
                          <Button 
                            className={classes.button + label.name} 
                            key = {label.key}
                            style={
                            {backgroundColor:label.backgroundColor,
                            color:label.textColor,
                            fontWeight:'bold',
                            margin:'5px',
                            }} 
                            size={"small"}
                          >
                            {label.name}
                          </Button>
                        </TableCell> : null]
                        ))}
                  </TableRow>
                  <TableRow key={"Precision"}>
                  {"Precision0" in this.props.statistics ? <TableCell>Precision </TableCell>: null}
                  {"Precision0" in this.props.statistics ? this.props.labelClasses.map(label => {
                    return (
                        label.name !== 'O'? [<TableCell align="right">{style(this.props.statistics["Precision"+label.key])}</TableCell>
                      ,this.statDelta("Precision"+label.key)] : null              
                    )
                  }) : null}
                  </TableRow>
                  <TableRow key={"Recall"}>
                  {"Recall0" in this.props.statistics ? <TableCell>Recall </TableCell>: null}
                  {"Recall0" in this.props.statistics ? this.props.labelClasses.map(label => {
                    return (
                      label.name !== "O"? [<TableCell align="right">{style(this.props.statistics["Recall"+label.key])}</TableCell>
                      ,this.statDelta("Recall"+label.key)] : null                    
                    )
                  }) : null}
                  </TableRow>
                  <TableRow key={"F1"}>
                  {"F10" in this.props.statistics ? <TableCell>F1 </TableCell>: null}
                  {"F10" in this.props.statistics ? this.props.labelClasses.map(label => {
                    return (
                      label.name !== "O"? [<TableCell align="right">{style(this.props.statistics["F1"+label.key])}</TableCell>
                      ,this.statDelta("F1"+label.key)] : null 
                    )
                  }) : null}
                  </TableRow>
              </TableBody>
            </Table>
          </Paper>        )
    }
}

function mapStateToProps(state, ownProps?) {
    return { 
      statistics: state.statistics_LRmodel.data,
      pending: state.statistics_LRmodel.pending,
      labelClasses: state.labelClasses.data
    };
}
function mapDispatchToProps(dispatch) {
    return {
      getLRStatistics: bindActionCreators(getLRStatistics, dispatch)
    };
}
export default connect(mapStateToProps, mapDispatchToProps)(CRFStatisticsPane);