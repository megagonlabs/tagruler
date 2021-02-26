import React from 'react';

import { styled, withStyles } from '@material-ui/core/styles';
import Badge from '@material-ui/core/Badge';
import Box from '@material-ui/core/Box';
import Tooltip from '@material-ui/core/Tooltip';

export const InlineBox = styled(Box)({
    width: "fit-content",
    display: "inline"
});

const StyledBadge = withStyles((theme) => ({
  badge: {
    right: "50%",
    top: 10,
    borderRadius: "50%",
    //background: `${theme.palette.background.paper}`,
    //padding: '0 4px',
  },
}))(Badge);

const InlineBadge = styled(StyledBadge)({
    width: "fit-content",
    display: "inline"
});

export function NERSpan(text, NER, start_offset) {
    return(
        <InlineBox 
            borderBottom={1} 
            key={`${start_offset}_NER`}
        >
        <Tooltip title={NER.label + ": " + NER.explanation}>
        <InlineBadge 
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          badgeContent={<span id={"annotation"}>{NER.label}</span>} //should be NER.label but that breaks the annotation...
        >{text}</InlineBadge>
        </Tooltip>
        </InlineBox>
    )
}

export function MatchedSpan(text, matched_span_data, start_offset) {
    return(
        <InlineBox 
            fontWeight="fontWeightBold" 
            key={`${start_offset}_match`}
        >
        {text}
        </InlineBox>
    )
}