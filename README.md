# San Diegus Clarifaicus the mystic storyteller

*He tells you the darnest stories you'll ever hear. What makes it better? He read its off your face.*

Stories are fun, and they usually require an in-depth creative process by an author. Ever wondered how a computer would write stories after it has read *Alice In Wonderland* and was given a picture to start off the beginning? In this project, produced during SDHacks 2018, we use Clarifai's image classification API to generate list of words(concepts) that describes the given picture, and run a LSTM RNN model to generate a text based on the list.

## The approach (with details)
Our initial approach consisted of Neural net consisting LSTM with Conv2D layers, and Dropout and BatchNorm layers for regularization.

In this case, we do not generate a validation dataset through validation split, as the network will benefit more from larger training data set, and will not benefit as much from testing loss/metric.

## Conclusion
something to wrap up this project and how good it is blah blah

## The files
* filename: explanation of the file, duh.
* filename: explanation of the file, duh.
* filename: explanation of the file, duh.
* filename: explanation of the file, duh.
* filename: explanation of the file, duh.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Caelen Wang** - [21wangc](https://github.com/21WANGC)
* **Mariam Gadzhimagomedova** - [mgG809](https://github.com/mgG809)
* **JoonHo (Brian) Lee** - [JHL0513](https://github.com/JHL0513)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to SDHacks and its hosting staff, and the sponsors of the event
* https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
* Thanks to Clarifai for the API

