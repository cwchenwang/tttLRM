// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

// Method configuration: method name -> index in the combined video
// Layout: Method1 | Method2 | Ours  (3 segments total)
var methodConfig = {
    'method1': 0,
    'method2': 1
};

var currentMethodIndex = 0;
var totalSegments = 3; // 2 baselines + Ours

function showComparison(target) {
    var elements = document.getElementsByClassName('tab')
    for (var i = 0; i < elements.length; i++) {
        if (elements[i].id.includes('compare')) {
            elements[i].className = 'tab'
        }
    }
    target.className = "tab active";

    var elements = document.getElementsByClassName('video-compare-container')
    for (var i = 0; i < elements.length; i++) {
        if (elements[i].id.includes('compare')) {
            if (elements[i].id.includes(target.id))
                elements[i].style.display = 'block'
            else
                elements[i].style.display = 'none'
        }
    }

    // For visible containers, ensure the video is loaded and playing
    var visibleContainers = document.querySelectorAll('.video-compare-container[style*="block"]');
    visibleContainers.forEach(function (container) {
        var video = container.querySelector('video');
        if (!video) return;

        // If metadata not ready, wait for it then start
        if (video.readyState < 1) {
            video.addEventListener('loadedmetadata', function handleLoaded() {
                video.removeEventListener('loadedmetadata', handleLoaded);
                resizeAndPlay(video);
            });
            video.load();
        } else {
            resizeAndPlay(video);
        }
    });
}

function selectMethod(element) {
    var method = element.getAttribute('data-method');

    // Update active state
    var methodTabs = document.querySelectorAll('.method-pill[data-method]');
    methodTabs.forEach(function (tab) {
        tab.classList.remove('active');
    });
    element.classList.add('active');

    // Change method
    changeMethod(method);
}

function changeMethod(method) {
    currentMethodIndex = methodConfig[method] || 0;

    // Update method-index attribute on all videos
    var videos = document.querySelectorAll('video[data-method-index]');
    videos.forEach(function (video) {
        video.setAttribute('data-method-index', currentMethodIndex);

        // Reload video to trigger resizeAndPlay
        var container = video.closest('.video-compare-container');
        if (container && container.style.display !== 'none') {
            video.load();
        }
    });
}

 

function setComparisonToggleLabel(isPlaying) {
    var toggle = document.getElementById('compare_toggle_play');
    if (!toggle) return;
    toggle.textContent = isPlaying ? '❚❚' : '▶';
}

function toggleComparisonPlayback() {
    var videos = document.querySelectorAll('video[data-method-index]');
    if (!videos.length) return;

    var shouldPause = Array.prototype.some.call(videos, function (video) {
        return !video.paused && !video.ended;
    });

    videos.forEach(function (video) {
        if (shouldPause) {
            video.pause();
        } else {
            video.play();
        }
    });

    setComparisonToggleLabel(!shouldPause);
}

// Initialize with default method on page load
document.addEventListener('DOMContentLoaded', function () {
    changeMethod('method1');
    // Ensure initial visible containers are initialized (video by default)
    var activeTab = document.getElementById('compare_video') || document.querySelector('.tab');
    if (activeTab) {
        showComparison(activeTab);
    } else {
        // No tabs: initialize visible containers directly
        var visibleContainers = document.querySelectorAll('.video-compare-container');
        visibleContainers.forEach(function (container) {
            if (container.style.display === 'none') return;
            var video = container.querySelector('video');
            if (video) {
                if (video.readyState < 1) {
                    video.addEventListener('loadedmetadata', function handleLoaded() {
                        video.removeEventListener('loadedmetadata', handleLoaded);
                        resizeAndPlay(video);
                    });
                    video.load();
                } else {
                    resizeAndPlay(video);
                }
            }
        });
    }
});


function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);
    var methodIndex = parseInt(vid.getAttribute('data-method-index')) || 0;

    // If video data not ready, wait for canplay then retry
    if (vid.readyState < 2) {
        var onCanPlay = function () {
            vid.removeEventListener('canplay', onCanPlay);
            playVids(videoId);
        };
        vid.addEventListener('canplay', onCanPlay);
        vid.load();
        return;
    }

    var position = 0.5;
    // Layout: Method1 | Method2 | Ours (3 segments total)
    // Select baseline by methodIndex, ours is fixed at last segment
    var segmentWidth = vid.videoWidth / totalSegments; // Each segment width
    var methodStartX = methodIndex * segmentWidth;     // Selected baseline segment
    var oursStartX = segmentWidth * (totalSegments - 1); // Ours at last segment

    // Canvas displays one segment width, showing MethodX on left and Ours on right via slider
    var vidWidth = segmentWidth;  // Display width (one segment)
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove", trackLocation, false);
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove", trackLocationTouch, false);


        function drawLoop() {
            // Draw baseline segment (left side) - full canvas width
            mergeContext.drawImage(vid, methodStartX, 0, segmentWidth, vidHeight, 0, 0, vidWidth, vidHeight);

            // Draw Ours segment (right side) based on slider position
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - colStart).clamp(0.0, vidWidth);
            var sourceColStart = oursStartX + (colStart / vidWidth) * segmentWidth;
            var sourceColWidth = (colWidth / vidWidth) * segmentWidth;

            mergeContext.drawImage(vid, sourceColStart, 0, sourceColWidth, vidHeight, colStart, 0, colWidth, vidHeight);

            requestAnimationFrame(drawLoop);


            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(currX, arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()

            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth * position, 0);
            mergeContext.lineTo(vidWidth * position, vidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = "#444444";
            mergeContext.lineWidth = 5;
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);

            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowWidth / 2);

            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY);

            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#444444";
            mergeContext.fill();



        }
        requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};


function resizeAndPlay(element) {
    var cv = document.getElementById(element.id + "Merge");
    // Calculate canvas width: display one segment width (for MethodX | Ours comparison)
    var segmentWidth = element.videoWidth / totalSegments;
    if (!segmentWidth || !isFinite(segmentWidth)) return;

    cv.width = segmentWidth;
    cv.height = element.videoHeight;
    element.play();
    setComparisonToggleLabel(true);
    element.style.height = "0px";  // Hide video without stopping it

    playVids(element.id);
}

