
RESOURCE_URL = 'http://127.0.0.1:5000/frame';
var count_upserver = 0;
function UpserverFrame(image, buffer){
    // console.log(image);
    count_upserver += 1;
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
}

function convertImgDataToUint8Array(imgdata){
    return Array.from(imgdata.data);
}

function convertImgDataToPNGURL(imgdata){
    convertURLctx.putImageData(imgdata, 0, 0);
    const base64Img = convertURLCanvas.toDataURL('image/png');
    // console.log(tempRCanvas);

    // let file = null;
    // let blob = tempCanvas.toBlob(function(blob) {
    //     file = new File([blob], 'frame.png', { type: 'image/png' });
    // }, 'image/png');
    return base64Img;
}

function ModelResponseHandler(response, buffer){
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
    }

    this.countLabel = 0;
    this.LabelFrames = function(action, indicesNeighbor){
        this.countLabel+=1;
        this.idNextProccessed = this.idLastProccessed + action;
        // 2 lines below may not be used
        this.listFrames[this.idNextProccessed].isProccessed = true;
        this.listFrames[this.idNextProccessed].isSelected = true;

        for(let idNeighbor of indicesNeighbor){
            this.listFrames[idNeighbor].isSelected = true;
        }
    }

    this.countCookie = 0;
    this.CookieFrame = function(image){
        this.countCookie += 1;
        this.listFrames[this.idPoint].image = image;
        this.idPoint += 1;

        if (this.idNextProccessed <= this.idPoint - 1){
            UpserverFrame(this.listFrames[this.idNextProccessed].image, this);
            // console.log(this.idNextProccessed, this.listFrames[this.idNextProccessed].image);
            // document.dispatchEvent(new CustomEvent('upserver', {'image':this.listFrames[this.idNextProccessed].image, 'buffer':this}));
            this.idLastProccessed = this.idNextProccessed;
            this.idNextProccessed = Infinity;
        }

        if (this.idPoint >= this.idMaxPoint){
            this.Expired();
        }
    }
}



// var savedFrames = []


// alert(Buffer)
// console.log(typeof(Buffer))
// console.log(Buffer)
