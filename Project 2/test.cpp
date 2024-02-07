#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_File_Chooser.H>

// Callback function for folder selection
void select_folder(Fl_Widget* widget, void* data) {
    Fl_File_Chooser chooser(".", "*", Fl_File_Chooser::DIRECTORY, "Select Folder");
    chooser.show();
    while (chooser.shown())
        Fl::wait();
    if (chooser.value() != NULL)
        static_cast<Fl_Input*>(data)->value(chooser.value());
}

// Callback function for file selection
void select_file(Fl_Widget* widget, void* data) {
    Fl_File_Chooser chooser(".", "*", Fl_File_Chooser::SINGLE, "Select File");
    chooser.show();
    while (chooser.shown())
        Fl::wait();
    if (chooser.value() != NULL)
        static_cast<Fl_Choice*>(data)->add(chooser.value());
}

// Callback function for exit button
void exit_program(Fl_Widget* widget, void*) {
    exit(0);
}

int main(int argc, char **argv) {
    Fl_Window *window = new Fl_Window(400, 200, "File Selector");

    Fl_Input *directoryInput = new Fl_Input(100, 20, 200, 30, "Directory:");
    Fl_Button *folderButton = new Fl_Button(310, 20, 70, 30, "Select");
    folderButton->callback(select_folder, directoryInput);

    Fl_Choice *fileChoice = new Fl_Choice(100, 70, 200, 30, "Files:");
    Fl_Button *fileButton = new Fl_Button(310, 70, 70, 30, "Select");
    fileButton->callback(select_file, fileChoice);

    Fl_Button *exitButton = new Fl_Button(150, 120, 100, 30, "Exit");
    exitButton->callback(exit_program);

    window->end();
    window->show(argc, argv);
    return Fl::run();
}
