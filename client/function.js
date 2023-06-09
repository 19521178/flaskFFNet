
RESOURCE_URL = 'http://127.0.0.1:5000/frame';
var cookieTimes = [];
var upserverTimes = [];
var convertPngTimes = [];
var labelTimes = [];
var expireTimes = [];


var count_upserver = 0;
function UpserverFrame(image, buffer){
    count_upserver += 1;
    // const data = image;
    const data = convertImgDataToPNGURL(image);
    // console.log(data);
    fetch(RESOURCE_URL, {
        method: 'POST',
        headers: {
            'Accept': 'application.json',
            // 'Content-Type': 'multipart/form-data'
            // 'Content-Type': 'image/png'
        },
        // body: JSON.stringify(image),
        body: data,
        // caches: 'no-cache'
    })
    .then(response => response.json())
    .then(data => ModelResponseHandler(data, buffer))
    .catch(error => console.log(error));
    delete data;
    // var data = {
    //     "action": 1,
    //     "indices neighbor": [0]
    // };
    // ModelResponseHandler(data, buffer);
    // // delete data;
}

function convertImgDataToUint8Array(imgdata){
    return Array.from(imgdata.data);
}

function convertImgDataToPNGURL(imgdata){
    var start_convertPng_time = Date.now();
    convertURLctx.putImageData(imgdata, 0, 0);
    const base64Img = convertURLCanvas.toDataURL('image/jpeg', quality=0.2);
    var end_convertPng_time = Date.now();
    convertPngTimes.push(end_convertPng_time-start_convertPng_time);
    // console.log(tempRCanvas);

    // let file = null;
    // let blob = tempCanvas.toBlob(function(blob) {
    //     file = new File([blob], 'frame.png', { type: 'image/png' });
    // }, 'image/png');
    return base64Img;
}

function ModelResponseHandler(response, buffer){
    // console.log(response);
    let action = response['action'];
    let indicesNeighbor = response['indices neighbor'];
    console.log(response)
    buffer.LabelFrames(action, indicesNeighbor);
}

function BufferFrame(){
    this.image = undefined;
    this.isSelected = false;
    this.isProccessed = false;
}

function Buffer(length, idMaxPoint, savedFrames){
    this.savedFrames = savedFrames;
    this.listFrames = [];
    for (let i=0; i<length; i++){
        this.listFrames.push(new BufferFrame());
    }
    this.idPoint = 0;
    this.idMaxPoint = idMaxPoint;
    this.idLastProccessed = -1;
    this.idNextProccessed = 0;

    this.countExpired = 0;
    this.Expired = function(){
        var start_expire_time = Date.now();
        // if (this.idNextProccessed >= 0){
            this.countExpired+=1;
            // Pop first element and push new init element to tail
            let expiredFrame = this.listFrames.shift();
            this.listFrames.push(new BufferFrame());
            
            this.idLastProccessed -= 1;
            this.idNextProccessed -= 1;
            this.idPoint -= 1;

            if (expiredFrame.isSelected === true){
                savedFrames.push(expiredFrame.image);
            }

            delete expiredFrame;
        // }
        var end_expire_time = Date.now();
        expireTimes.push(end_expire_time-start_expire_time);
        
    }

    this.countLabel = 0;
    this.LabelFrames = function(action, indicesNeighbor){
        // console.log(action);
        // console.log("before: ", this.idLastProccessed, this.idNextProccessed);
        this.countLabel+=1;
        var start_label_time = Date.now();
        this.idNextProccessed = this.idLastProccessed + action;
        // 2 lines below may not be used
        this.listFrames[this.idNextProccessed].isProccessed = true;
        this.listFrames[this.idNextProccessed].isSelected = true;

        for(let idNeighbor of indicesNeighbor){
            this.listFrames[idNeighbor].isSelected = true;
        }
        var end_label_time = Date.now();
        labelTimes.push(end_label_time-start_label_time);
        console.log(this.idLastProccessed, this.idNextProccessed);
    }

    this.countCookie = 0;
    this.CookieFrame = function(image){
        this.countCookie += 1;
        console.log('Count cookie: ' + this.countCookie);
        var start_cookie_time = Date.now();
        this.listFrames[this.idPoint].image = image;
        this.idPoint += 1;

        if (this.idNextProccessed <= this.idPoint - 1){
            this.idLastProccessed = this.idNextProccessed;
            this.idNextProccessed = Infinity;
            var start_upserver_time = Date.now();
            UpserverFrame(this.listFrames[this.idLastProccessed].image, this);
            var end_upserver_time = Date.now();
            upserverTimes.push(end_upserver_time - start_upserver_time);
            // console.log(this.idNextProccessed, this.listFrames[this.idNextProccessed].image);

        }

        if (this.idPoint >= this.idMaxPoint){
            this.Expired();
        }
        var end_cookie_time = Date.now();
        cookieTimes.push(end_cookie_time - start_cookie_time);
        
    }
}



// var savedFrames = []


// alert(Buffer)
// console.log(typeof(Buffer))
// console.log(Buffer)
// totalCookieTimes = [];
// for (let i = 0; i < cookieTimes.length; i++){
//     totalCookieTimes.push(getImgDataTimes[i] + cookieTimes[i])
// }