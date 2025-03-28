// Define the AngularJS Application
const app = angular.module('CardGridApp', []);

// Create a Controller
app.controller('CardController', function ($scope) {
    $scope.cards = [
        { title: "Card Title 1" },
        { title: "Card Title 2" },
        { title: "Card Title 3" },
        { title: "Card Title 4" },
        { title: "Card Title 5" },
        { title: "Card Title 6" },
        { title: "Card Title 7" },
        { title: "Card Title 8" },
        { title: "Card Title 9" },
        { title: "Card Title 10" },
    ];

    $scope.viewTrials = function (title) {
        alert(`Viewing trials for ${title}`);
    };
});
