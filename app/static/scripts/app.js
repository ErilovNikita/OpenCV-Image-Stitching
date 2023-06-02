function genParoram( 
    file1,
    file2
) {
    return new Promise(function (resolve, reject) {
        var url = `/api/gereration`

        let xhr = new XMLHttpRequest();
        xhr.open("POST", url);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };

        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send( JSON.stringify({
            "file1" : file1,
            "file2" : file2,
        }) );
    });
}