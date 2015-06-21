let assert = (condition, message="Assertion failed") => {
  if (condition)
		return;
	
  throw new Error(message);
}

// helper function returns array of zeros of length n
// and uses typed arrays if available
let zeros = (n) => {
  assert(!isNaN(n));

  if(typeof ArrayBuffer === 'undefined') {
    // lacking browser support
    var arr = new Array(n);
    for(var i=0;i<n;i++) { arr[i] = 0; }
    return arr;
  } else {
    return new Float64Array(n);
  }
}

export default {
	assert,
	zeros
};