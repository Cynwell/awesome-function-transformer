//// /assets/copy.js
document.addEventListener('click', function(event) {
    if(event.target && event.target.nodeName === "BUTTON" && event.target.hasAttribute("data-equation")) {
        const eq = event.target.getAttribute("data-equation");
        navigator.clipboard.writeText(eq).then(function() {
            // Optionally, provide user feedback
            console.log("Copied: " + eq);
        }, function(err) {
            console.error("Could not copy text: ", err);
        });
    }
});