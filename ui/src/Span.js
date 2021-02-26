let Span = function(startOffset, endOffset, text, label=null) {
    const span = {
        id: startOffset,
        label: label,
        start_offset: startOffset,
        end_offset: endOffset,
        text: text.slice(startOffset, endOffset),
        link: null,
        spanAnnotated: false,
        spanLabel: null,
        isPositive: true
    };
    return span;
}

export default Span;