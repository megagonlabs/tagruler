import React from 'react';
import {connect} from "react-redux";

import ArrowDropUpIcon from '@material-ui/icons/ArrowDropUp';
import ArrowDropDownIcon from '@material-ui/icons/ArrowDropDown';
import Divider from '@material-ui/core/Divider';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';

import { style } from './SortingTableUtils'


class StatisticsPane extends React.Component {
    componentDidUpdate(prevProps, prevState) {
      if (Object.keys(prevProps.statistics).length > 0) {
        if (!this.statsSame(prevProps.statistics, this.props.statistics)) {
          this.setState({prevStats: prevProps.statistics});
        }
      }
    }

    statsSame(oldStats, newStats) {
      const stat_list = Object.keys(oldStats);
      for (var i = stat_list.length - 1; i >= 0; i--) {
        let stat_key = stat_list[i];
        if (oldStats[stat_key] !== newStats[stat_key]) {
          return false;
        }
      } return true;
    }

    statDelta(key) {
      if ((this.state) && ("prevStats" in this.state)) {
        const delta = this.props.statistics[key] - this.state.prevStats[key];
        let cellContent = delta;
        if (delta > 0){
          cellContent = <span style={{color:"green"}}><ArrowDropUpIcon/> {style(delta)} </span>
        } else if (delta < 0) {
          cellContent = <span style={{color:"red"}}><ArrowDropDownIcon/> {style(delta)}</span>
        } 
        return(<TableCell>{cellContent}</TableCell>)
      }
    }

    render(){
        const classes = this.props.classes;

        const statNameMap = {
          'f1': 'F1-score',
          'precision': 'Precision',
          'recall': 'Recall',
          'label_coverage': 'Coverage'
        }
        return(
          <Paper className={classes.paper}>
          
            <Typography className={classes.title} variant="h6" id="tableTitle">
                Labeling Statistics
            </Typography>
            <Typography variant="body1">Your labeling functions are agreggated by a labeling model then evaluated on the labeled development set. More information about these metrics is available via <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics">Scikit-Learn</a>.</Typography>

            {this.props.pending ? <LinearProgress/> : <Divider/>}

            <Table stickyHeader className={classes.table} size="small" aria-label="labeling statistics">
              <TableBody>
                {Object.keys(this.props.statistics).map(key => 
                    <TableRow key={key}>
                        <TableCell>{statNameMap[key]}</TableCell>
                        <TableCell align="right">{style(this.props.statistics[key])}</TableCell>
                        { this.statDelta(key) }
                    </TableRow>)}
              </TableBody>
            </Table>
          </Paper>        )
    }
}

function mapStateToProps(state, ownProps?) {
    return { 
      statistics: state.statistics.data,
      pending: state.statistics.pending
    };
}
function mapDispatchToProps(dispatch) {
    return {};
}
export default connect(mapStateToProps, mapDispatchToProps)(StatisticsPane);