import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { CommonModule } from '@angular/common'; // Import this

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    CommonModule // Add this to the imports array
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
