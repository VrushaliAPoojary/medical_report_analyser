import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  cards = [
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

  viewTrials(title: string): void {
    alert(`Viewing trials for ${title}`);
  }
}
