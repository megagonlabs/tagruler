import React from 'react';
import { createMuiTheme } from '@material-ui/core/styles';
import { makeStyles } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import ColorPalette from  './ColorPalette';
import Main from './Main';

const windowHeight = Math.max(
    document.documentElement.clientHeight,
    window.innerHeight || 0);

const useStyles = makeStyles(theme => ({
    root: {
        width: '100%',
        height: windowHeight,
    }
}));


const App = () => {
    const  theme = createMuiTheme(ColorPalette),
        classes = useStyles();

    return (

        <ThemeProvider theme={theme}>
            <div className={classes.root}>
                <Main/>
            </div>
        </ThemeProvider>
    );
};

export default App;
