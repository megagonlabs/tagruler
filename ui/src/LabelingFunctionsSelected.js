import React from 'react';
import Button from '@material-ui/core/Button';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";
import { withStyles } from '@material-ui/core/styles';


import ClearIcon from '@material-ui/icons/Clear';
import Divider from '@material-ui/core/Divider';
import IconButton from '@material-ui/core/IconButton';
import LinearProgress from '@material-ui/core/LinearProgress';
import Paper from '@material-ui/core/Paper';
import PublishIcon from '@material-ui/icons/Publish';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import Tooltip from '@material-ui/core/Tooltip';
import Typography from '@material-ui/core/Typography';
import WarningIcon from '@material-ui/icons/Warning';

import LFPanel from './LFPanel'

import submitLFs, { getLFstats, deleteLF } from './actions/submitLFs'
import { stableSort, getSorting, style } from './SortingTableUtils'

const FadedTableBody = withStyles(theme => ({
  root: {
    color: "white",
  },
}))(TableBody);

class LabelingFunctions extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            order: 'asc',
            orderBy: "ID"
        };
    }

    componentDidMount() {
        this.props.getLFstats();
    }

    label(lf) {
        const label = this.props.labelClasses.filter(c => c.key === lf.Label)
        return (
            label[0] ? label[0].name : "UNK"
        );
    }


    conditionToString(condition) {
        let string = condition["string"];
        if (condition["case_sensitive"]) {
            string = "<b>"+string+"</b>";
        }
        if (condition.type === this.props.keyType["CONTEXT_SIMILAR"]){
            string = "\""+string +"\""
        }

        let condition_type_text = <span>{condition.TYPE_}</span>
        if ("explanation" in condition) { // Add mouseover explanation, if available
            condition_type_text = <Tooltip title={condition.explanation} aria-label={condition.explanation}>{condition_type_text}</Tooltip>
        } 

        if (condition.positive) {
            return <>{string} ({condition_type_text})</>;
        } else {
            return <>NOT({string}) ({condition_type_text})</>;
        }
    }

    conditions(lf) {
        const conditions = lf.Conditions.map(cond => this.conditionToString(cond));
        return conditions
            .reduce((prev, curr) => [prev, " " + lf.CONNECTIVE + " ", curr])
    }

    LFtoStrings(key, lf) {
        let warning = this.duplicateCell(lf.Duplicate);

        const ACC_THRESHOLD = 0.3;
        if (lf["Emp. Acc."] < ACC_THRESHOLD) {
            if (lf['Incorrect'] > 0) {
                warning = <Tooltip title={"low accuracy"}><WarningIcon color={"error"}/></Tooltip>
            }
        }
        const stringsDict = {
            //active: lf.active,
            id: key,
            conditions: this.conditions(lf),
            context: lf.CONTEXT_,
            label: this.label(lf),
            order: lf.Direction.toString(),
            weight: lf.Weight,
            warning: warning
        };
        return {...lf, ...stringsDict};
    }

    duplicateCell(dupeString) {
        if (dupeString !== null) {
            let msg = "labeling signature same as ".concat(dupeString)
            return(
                <Tooltip title={msg}><WarningIcon color={"error"}/></Tooltip>
            )
        }
    }

    render() {

        var headCells = [
          { id: 'ID', numeric: false, disablePadding: true, label: 'T' },
          { id: 'conditions', numeric: false, disablePadding: true, label: 'Conditions' },
          { id: 'label', numeric: false, disablePadding: true, label: 'Label' },
          { id: 'Emp. Acc.', numeric: true, disablePadding: true, label: 'Precision' },
          { id: 'Recall', numeric: true, disablePadding: true, label: 'Recall' },
          { id: 'Coverage Training', numeric: true, disablePadding: true, label: 'Train. Coverage' },
          { id: 'Conflicts Training', numeric: true, disablePadding: true, label: 'Train. Conflicts' },
        ];

        const { classes } = this.props;
        const createSortHandler = property => event => {
            const isDesc = orderBy === property && order === 'desc';
            this.setState({
                order: isDesc ? 'asc' : 'desc',
                orderBy: property
            });
        };

        //if (this.state.orderBy === "ID") {
        //    if (this.state.order === 'asc') {
        //        headCells[0]["label"] = "Oldest"
        //    } else {
        //        headCells[0]["label"] = "Newest"
        //    }
        //}

        const order = this.state.order;
        const orderBy = this.state.orderBy;

        var LFList = Object.keys(this.props.labelingFunctions).map(lf_key => this.LFtoStrings(lf_key, this.props.labelingFunctions[lf_key]));
        
        const inactive = LFList.filter(LF => !LF.active);
        LFList = LFList.filter(LF => LF.active);

        // Check if there are any LF warnings
        let any_warnings = false;
        if (LFList.some(lf => lf["warning"])) {
            any_warnings = true;
            /*headCells.push({ id: 'Warning', numeric: false, disablePadding: true, label: 'Warning' })*/
        }

        let any_context = false;
        if (LFList.some(lf => lf["context"])) {
            any_context = true;
            headCells.splice(2, 0, { id: 'CONTEXT_', numeric: false, disablePadding: true, label: 'Context'});
        }
        
        return(
          <Paper className={classes.paper}>
            <Typography className={classes.title} variant="h6" id="tableTitle">
                Selected Labeling Functions ({LFList.length}) 
            </Typography>
            {/*any_warnings ? <Typography color="error"><WarningIcon/>Some of your functions have duplicate labeling signatures or low accuracy. Consider deleting the indicated functions.</Typography> : ""}
            {this.props.pending ? <LinearProgress/> : <Divider/>*/}
            <Table size="small" aria-label="Selected labeling functions table">
              <TableHead>
                  <TableRow>
                    {headCells.map(headCell => (

                      <TableCell
                        key={headCell.id}
                        align={headCell.numeric ? 'right' : 'left'}
                        padding="none"
                        sortDirection={orderBy === headCell.id ? order : false}
                      ><TableSortLabel
                          active={orderBy === headCell.id}
                          direction={order}
                          onClick={createSortHandler(headCell.id)}
                        >{headCell.label}
                          {orderBy === headCell.id ? (
                            <span className={classes.visuallyHidden}>
                              {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                            </span>
                          ) : null}
                        </TableSortLabel>
                      </TableCell>

                    ))}
                    <TableCell></TableCell>
                  </TableRow>
              </TableHead>
              <TableBody>

                {stableSort(LFList, getSorting(order, orderBy))
                    .map(row => (
                  <TableRow key={row.id}>
                    <TableCell padding="none"><LFPanel {...this.props} lf={row}/></TableCell>
                    <TableCell padding="none">{row.conditions}</TableCell>
                    {any_context ? <TableCell align="right" padding="none">{row.context}</TableCell> : null }
                    
                    <TableCell align="center">
                        <Button 
                            className={classes.button + this.props.labelClasses.filter(l => l.name === row.label)[0].name} 
                            key = {this.props.labelClasses.filter(l => l.name === row.label)[0].key}
                            style={
                            {backgroundColor:this.props.labelClasses.filter(l => l.name === row.label)[0].backgroundColor,
                            color:this.props.labelClasses.filter(l => l.name === row.label)[0].textColor,
                            fontWeight:'bold',
                            margin:'5px',
                            }} 
                            size={"small"}
                        >
                            {this.props.labelClasses.filter(l => l.name === row.label)[0].name}
                        </Button>
                    </TableCell>
                    <TableCell padding="none" align="right">{"Emp. Acc." in row ? (style(row["Emp. Acc."])) : ("...")}</TableCell> 
                    <TableCell padding="none" align="right">{"Recall" in row ? (style(row["Recall"])) : ("...")}</TableCell> 
                    <TableCell padding="none" align="right">{"Coverage Training" in row ? (style(row["Coverage Training"])) : ("...")}</TableCell> 
                    <TableCell  padding="none"align="right">{"Conflicts Training" in row ? (style(row["Conflicts Training"])) : ("...")}</TableCell> 
                    {/*any_warnings ? (<TableCell align="right">{row["warning"]}</TableCell>) : null*/}                     
                    <TableCell padding="none">
                        <IconButton onClick={() => this.props.deleteLF([row.id])} size="small">
                            <ClearIcon/>
                        </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
            </Table>

            { inactive.length > 0 ? <Typography className={classes.title} variant="h6" id="tableTitle">Inactive Functions ({inactive.length})</Typography> : ""}
            <Table>
            <FadedTableBody>
               {inactive.map(row => (
                  <TableRow key={row.id}>
                    <TableCell padding="none"></TableCell>
                    <TableCell padding="none">{row.conditions}</TableCell>
                    {any_context ? <TableCell/>: null}
                    {/*<TableCell padding="none">{row.label}</TableCell>*/}
                    <TableCell align="center">
                        <Button 
                            className={classes.button + this.props.labelClasses.filter(l => l.name === row.label)[0].name} 
                            key = {this.props.labelClasses.filter(l => l.name === row.label)[0].key}
                            style={
                            {backgroundColor:this.props.labelClasses.filter(l => l.name === row.label)[0].backgroundColor,
                            color:this.props.labelClasses.filter(l => l.name === row.label)[0].textColor,
                            fontWeight:'bold',
                            margin:'5px',
                            }} 
                            size={"small"}
                        >
                            {this.props.labelClasses.filter(l => l.name === row.label)[0].name}
                        </Button>
                    </TableCell>
                    <TableCell align="right">{"dev_set_accuracy" in row    ? style(row.dev_set_accuracy)    : "..."}</TableCell> 
                    <TableCell align="right">{"Coverage" in row  ? style(row.Coverage)  : "..."}</TableCell> 
                    <TableCell align="right">{"Conflicts" in row ? style(row.Conflicts) : "..."}</TableCell> 
                    {any_warnings ? <TableCell/>: null}
                    <TableCell><IconButton onClick={
                        () => this.props.submitLFs({[row.id]: this.props.labelingFunctions[row.id]})
                    } size="small"><PublishIcon/></IconButton></TableCell>
                  </TableRow>
                ))}

            </FadedTableBody>
            </Table>
          </Paper>
        );
    }
}

function mapStateToProps(state) {
    return { 
        labelingFunctions: state.selectedLF.data,
        pending: state.selectedLF.pending,
        labelClasses: state.labelClasses.data,
        keyType: state.gll.keyType,
        stats: state.selectedLF.stats //to force update when statistics come in
    };
}

function mapDispatchToProps(dispatch) {
    return {
        getLFstats: bindActionCreators(getLFstats, dispatch),
        deleteLF: bindActionCreators(deleteLF, dispatch),
        submitLFs: bindActionCreators(submitLFs, dispatch)
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(LabelingFunctions);